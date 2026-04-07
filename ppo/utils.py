import matplotlib.pyplot as plt

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		print('Wrong Input.')
		raise
	
def plot_training_score(scores, steps, filename='training_score.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scores, marker='o', linestyle='-', color='b')
    plt.title('Training Score Over Time')
    plt.xlabel('Total Steps')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	return  2*(a-0.5)*max_action

def Reward_adapter(r, EnvIdex):
	# For BipedalWalker
	if EnvIdex == 0 or EnvIdex == 1:
		if r <= -100: r = -1
	# For Pendulum-v0
	elif EnvIdex == 3:
		r = (r + 8) / 8
	return r

def evaluate_policy(env, agent, max_action, turns):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			s_next, r, dw, tr, info = env.step(act)
			done = (dw or tr)

			total_scores += r
			s = s_next

	return total_scores/turns