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

from utils import str2bool, plot_training_score, Action_adapter, Reward_adapter, evaluate_policy

# Practical Loop
# PPO training loop
# Step 0: Initialization
# Actor parameters \theta
# Critic parameters \phi

# Step 1: Collect rollout
# For t=0,...,T-1 in env
# observe s_t
# Compute \pi(\dot|s_t) (policy distribution)
# Sample action a_t
# Compute log prob log(\pi(a_t|s_t))
# Compute value estimate V_{\phi}(s_t)
# Step env with a_t
# get r_t, s_t+1, done flag d_t and store
# if env terminates reset and continue collecting so rollout tensor remains full

# Step 2: get bootstrap value for final states
# After last rollout step compute V_{\phi}(S_T) for each env's final next-state

# Step 3: Compute GAE advantages
# Go backward through rollout and compute
# \delta_t=r_t+\gamma(1-d_t)*V_{\phi}(s_{t+1})-V_{\phi}}(s_t)
# \hat{A} = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1}
# Store all \hat{A}_t

# Step 4: Compute return targets
# \hat{R}_t = \hat{A}_t+V_\phi(s_t)
# Store all \hat{R}_t

# Step 5: Normalize advantages
# Normalize \hat{A}_t over the whole batch

# Step 6: Perform PPO Updates
# for K epochs:
#   shuffle full rollout batch
#   divide into minibatches
#   for each minibatch
# recompute current policy log probs log\pi_\thest(a_t|s_t)
# recompute current values V_\phi(s_t)
# Compute ratio
# r_t(\theta) = exp(log(\pi_\theta(a_t|s_t))-log(\pi_\theta_old(a_t|s_t))))
# Compute clipped actor objective
# Compute entropy bonus
# Form total loss
# Zero gradients
# Backpropagate
# clip gradient norm (optional)
# optimizer step

# Step 7: discard old rollout and collect new one
# once all K epochs are done, throw away old rollout batch
# Collect new data on updated policy
# Repeat forever

class Critic(nn.Module):
	def __init__(self, s_size , hidden):
		super(Critic, self).__init__()
		
        # neural network architecture for the critic
		self.layer1 = nn.Linear(s_size, hidden)
		self.layer2 = nn.Linear(hidden,hidden)
		self.layer3 = nn.Linear(hidden,1)
	
		self.apply(self._init_weights)

    # initialising the weights of the network
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				module.bias.data.zero_()

    # forward pass through the critic network
	def forward(self, s ):
		v = F.relu(self.layer1(s))
		v = F.relu(self.layer2(v))
		v = F.relu(self.layer3(v))
		return v

# actor with gaussian distribution (continuous action space)
class Actor(nn.Module):
	def __init__(self, s_size , a_size , hidden ):
		super(Actor, self).__init__()
		
		self.mulayer1 = nn.Linear(s_size, hidden)
		self.mulayer2 = nn.Linear(hidden, hidden)
		self.mulayer3 = nn.Linear(hidden, a_size)

		self.siglayer1 = nn.Linear(s_size, hidden)
		self.siglayer2 = nn.Linear(hidden, hidden)
		self.siglayer3 = nn.Linear(hidden, a_size)
		
		self.apply(self._init_weights)

	def forward(self, state):
		mu = F.relu(self.mulayer1(state))
		mu = F.relu(self.mulayer2(mu))
		mu = F.relu(self.mulayer3(mu))

		sigma = F.relu(self.siglayer1(state))
		sigma = F.relu(self.siglayer2(sigma))
		sigma = F.relu(self.siglayer3(sigma))
	
		return mu, sigma

	def get_dist(self, state):
		mu, sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu = F.relu(self.mulayer1(state))
		mu = F.relu(self.mulayer2(mu))
		mu = F.relu(self.mulayer3(mu))
		return mu

class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		
		state_size = 8
		action_size = 2
		hidden_size = 64
		buffer_length = 2048

		''' Build Actor '''
		self.actor = Actor(state_size, action_size, hidden_size)
		
		'''Build Critic'''
		self.critic = Critic(state_size, hidden_size)
		
		'''Build Replay Buffer'''
		self.state_buffer = np.zeros((state_size, buffer_length))
		self.action_buffer = np.zeros((action_size, buffer_length))
		self.next_state_buffer = np.zeros((state_size, buffer_length))
		self.log_prob_buffer = np.zeros((action_size, buffer_length))
		self.value_estimate_buffer = np.zeros((1, buffer_length))
		self.reward_buffer = np.zeros((1,buffer_length))
		self.done_buffer = np.zeros((1, buffer_length))


	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy. Making the performance more stable
				action = self.actor.deterministic_act(state)
				return action
			else:
				# only used when interact with the env
				[mu, sigma] = self.actor.forward(state)
				### ADD NORMAL DISTRIBUTION SAMPLING ###
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
		a_optim_iter_num = 
		c_optim_iter_num = 
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
				

			'''update the critic'''
			for i in range(c_optim_iter_num):
				'''get the batch data'''
				
				
				'''calculate critic loss'''
				

				'''apply L2 regularisation ONLY on weights'''
				
						
				'''backprop and optimise'''
				

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
    
    if opt.Loadmodel: 

    '''Run Training loop or Eval'''
    if opt.render:
        
    else:
        traj_lenth, total_steps = 0, 0
        while :
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