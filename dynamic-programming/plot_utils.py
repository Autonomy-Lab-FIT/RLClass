import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_values(V):
	# reshape value function
	clear_output(wait=True)

	V_sq = np.reshape(V, (4,4))

	# plot the state-value function
	plt.figure(figsize=(6,6), facecolor='none')
	im = plt.imshow(V_sq, cmap='cool')
	for (j,i),label in np.ndenumerate(V_sq):
		plt.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
	plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
	plt.axis('off')
	plt.margins(0)	
	plt.title('State-Value Function', color="white")
	plt.show()
	plt.pause(0.1)

def plot_policy(V,policy):
	V_sq = np.reshape(V, (4,4))

	# plot the state-value function
	plt.figure(figsize=(6,6), facecolor='none')
	im = plt.imshow(V_sq, cmap='cool')
	for (j,i),label in np.ndenumerate(V_sq):
		if np.nonzero(policy[4*i+j,:])[0] == 0:
			arrow = "\u2190" # LEFT
		elif np.nonzero(policy[4*i+j,:])[0] == 1:
			arrow = "\u2193" # DOWN
		elif np.nonzero(policy[4*i+j,:])[0] == 2:
			arrow = "\u2192" # RIGHT
		elif np.nonzero(policy[4*i+j,:])[0] == 3:
			arrow = "\u2191" # UP
		plt.text(i, j, arrow, ha='center', va='center', fontsize=14)
	plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
	plt.axis('off')
	plt.margins(0)
	plt.title('Optimal Policy', color="white")
	plt.show()