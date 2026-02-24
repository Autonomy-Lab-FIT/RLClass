import numpy as np
import matplotlib.pyplot as plt


def plot_values(V, size, im_size):
	# reshape the state-value function
	V = np.reshape(V, size)
	# plot the state-value function
	fig= plt.figure(figsize=im_size, facecolor='none')
	
	im = plt.imshow(V, cmap='cool')
	for (j,i),label in np.ndenumerate(V):
		plt.text(i, j, np.round(label,3), ha='center', va='center', fontsize=14)
	plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
	plt.title('State-Value Function', color="white")
	plt.show()

def plot_policy(V,policy, size, im_size):
	V = np.reshape(V, size)

	# plot the state-value function
	plt.figure(figsize=im_size, facecolor='none')
	im = plt.imshow(V, cmap='cool')
	for (j,i),label in np.ndenumerate(V):
		if policy[j,i] == 0:
			arrow = "\u2191" # UP
		elif policy[j,i] == 1:
			arrow = "\u2192" # RIGHT
		elif policy[j,i] == 2:
			arrow = "\u2193" # DOWN
		elif policy[j,i] == 3:
			arrow = "\u2190" # LEFT
		else: arrow = "\u2022"
		plt.text(i, j, arrow, ha='center', va='center', fontsize=14)
	plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
	plt.title('Optimal Policy', color="white")
	plt.show()