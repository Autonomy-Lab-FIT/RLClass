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

from utils import str2bool, Action_adapter, Reward_adapter, evaluate_policy

class Critic(nn.Module):
	def __init__(self, _ , _):
		super(Critic, self).__init__()
		
        # neural network architecture for the critic
		
	
		self.apply(self._init_weights)

    # initialising the weights of the network
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				module.bias.data.zero_()

    # forward pass through the critic network
	def forward(self, _ ):
		v = _ 
		return v

# actor with gaussian distribution (continuous action space)
class Actor(nn.Module):
	def __init__(self, _ , _ , _ ):
		super(Actor, self).__init__()
		
		

	def forward(self, _ ):
		mu = torch.sigmoid( _ )
		sigma = F.softplus( _ )
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
		
		''' Build Actor '''
		self.actor = _ 
		self.actor_optimizer = _ 
		
		'''Build Critic'''
		self.critic = _ 
		self.critic_optimizer = _ 
		
		'''Build Replay Buffer'''


	def select_action(self, _ , deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy. Making the performance more stable
				
			
				return 
			else:
				# only used when interact with the env
				
				
				return 


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
			'''calculate values at current and next timestep'''
			

			'''dw for TD_target and Adv'''
			

			'''done for GAE'''
			


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):
			
			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			# loop through all the batches of data
			for i in range(a_optim_iter_num):
				'''get the batch data'''
				

				'''calculate log probablity ratio'''
				


				'''calculate CLIP loss parts'''
				


				'''backprop and optimise'''
				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				'''get the batch data'''
				
				
				'''calculate critic loss'''
				

				'''apply L2 regularisation ONLY on weights'''
				
						
				'''backprop and optimise'''
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
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
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
    EnvName = []
    BrifEnvName = []

    '''Build Env'''
       

    # print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
        #   '  max_a:',opt.max_action,'  min_a:',env.action_space.low[0], 'max_steps', opt.max_steps)

    '''Seed Everything'''
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    '''Use plotting to record training curves'''
    scores_list = []
    steps_list = []
    plot_filename = 'logs/training_score.png'
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    

    '''Create PPO agent and directory to store'''
    if not os.path.exists('model'): os.mkdir('model')
    
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    '''Run Training loop or Eval'''
    if opt.render:
        
    else:
        traj_lenth, total_steps = 0, 0
        while total_steps < opt.Max_train_steps:
			'''Reset Env and initialise'''
			'''Interact & train'''
			while not done:
				'''Interact with Env'''
				

				'''Store the current transition'''
				

				'''Fill initial entry of scores'''
				
				
				'''Update if its time'''
				
				
				'''Record & log'''
				
				
				'''Save model'''
                


if __name__ == '__main__':
    main()