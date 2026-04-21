import numpy as np
import copy
import math
from datetime import datetime
import os, shutil
import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

# from collections import namedtuple
# import random

from utils import str2bool, plot_training_score, Action_adapter, Reward_adapter, evaluate_policy

class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()
		
        # neural network architecture for the critic
		self.critic = nn.Sequential(
			nn.Linear(state_dim, net_width),
			nn.Tanh(),
			nn.Linear(net_width, net_width),
			nn.Tanh(),
			nn.Linear(net_width, 1),
        )
	
		self.apply(self._init_weights)

    # initialising the weights of the network
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				module.bias.data.zero_()

    # forward pass through the critic network
	def forward(self, state):
		v = self.critic(state)
		return v

# actor with gaussian distribution (continuous action space)
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Actor, self).__init__()
		
		self.mu_head = nn.Sequential(
			nn.Linear(state_dim, net_width),
			nn.Tanh(),
			nn.Linear(net_width, net_width),
			nn.Tanh(),
			nn.Linear(net_width, action_dim)
        )

		self.logstd = nn.Parameter(torch.zeros(action_dim))  # learnable log σ

	def forward(self, state):
		mu = self.mu_head(state)         # unbounded
		sigma = torch.exp(self.logstd)   # stable, constant per action dim
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, _ = self.forward(state)
		return mu

class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

        # Build Actor
		self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Replay buffer
		self.s_buffer = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.a_buffer = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.r_buffer = np.zeros((self.T_horizon, 1),dtype=np.float32)
		self.s_next_buffer = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.logprob_a_buffer = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.done_buffer = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		self.dw_buffer = np.zeros((self.T_horizon, 1),dtype=np.bool_)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy. Making the performance more stable
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)


	def train(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''						
		s = torch.from_numpy(self.s_buffer).to(self.dvc)
		a = torch.from_numpy(self.a_buffer).to(self.dvc)
		r = torch.from_numpy(self.r_buffer).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_buffer).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_buffer).to(self.dvc)
		done = torch.from_numpy(self.done_buffer).to(self.dvc)
		dw = torch.from_numpy(self.dw_buffer).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (~dw) - vs # for the states that are not done
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			# calculate GAE backwards - with [::-1] indexing
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
				adv.append(advantage)
			adv.reverse() # reverse to original order
			adv = copy.deepcopy(adv[0:-1]) # remove extra zero
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):
			
			# batch = self.buffer.sample(self.a_optim_batch_size)

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			# loop through all the batches of data
			for i in range(a_optim_iter_num):
				# get the specific batches
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				s_batch = s[index]
				a_batch = a[index]
				logprob_a_batch = logprob_a[index]
				adv_batch = adv[index]
				
				distribution = self.actor.get_dist(s_batch)
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a_batch)
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a_batch.sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv_batch
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv_batch
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				td_target_batch = td_target[index]
				s_batch = s[index]
				c_loss = (self.critic(s_batch) - td_target_batch).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
		self.s_buffer[idx] = s
		self.a_buffer[idx] = a
		self.r_buffer[idx] = r
		self.s_next_buffer[idx] = s_next
		self.logprob_a_buffer[idx] = logprob_a
		self.done_buffer[idx] = done
		self.dw_buffer[idx] = dw

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))



'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(2.5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def plot_training_score(scores, steps, filename='training_score.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scores, marker='o', linestyle='-', color='b')
    plt.title('Training Score Over Time')
    plt.xlabel('Total Steps')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
	
    # Build Env
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_steps = env._max_episode_steps
    print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
          '  max_a:',opt.max_action,'  min_a:',env.action_space.low[0], 'max_steps', opt.max_steps)

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Use plotting to record training curves
    scores_list = []
    steps_list = []
    plot_filename = 'logs/training_score.png'
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    

    # Create PPO agent and directory to store
    if not os.path.exists('model'): os.mkdir('model')
    agent = PPO_agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, 1)
            print(f'Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}')
    else:
        traj_lenth, total_steps = 0, 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                '''Interact with Env'''
                a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
                act = Action_adapter(a,opt.max_action) #[0,1] to [-max,max]
                s_next, r, dw, tr, info = env.step(act) # dw: dead&win; tr: truncated
                r = Reward_adapter(r, opt.EnvIdex)
                done = (dw or tr)

                '''Store the current transition'''
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                s = s_next
				
                '''Fill initial entry of scores'''
                if total_steps == 0:
                    steps_list.append(total_steps)
                    score = evaluate_policy(eval_env, agent, opt.max_action, turns=3)
                    scores_list.append(score)
					

                traj_lenth += 1
                total_steps += 1

                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, opt.max_action, turns=3) # evaluate the policy for 3 times, and get averaged result
                    scores_list.append(score)
                    steps_list.append(total_steps)
                    plot_training_score(scores_list, steps_list, plot_filename)
                    print('EnvName:',EnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))

        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()