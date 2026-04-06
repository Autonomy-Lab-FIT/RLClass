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