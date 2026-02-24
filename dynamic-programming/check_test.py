import unittest
import copy
from IPython.display import Markdown, display
import numpy as np
import gymnasium as gym

def printmd(string):
    display(Markdown(string))

def policy_evaluation_soln(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

def q_from_v_soln(env, V, s, gamma=1):
    q = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        for prob, next_state, reward, done in env.unwrapped.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement_soln(env, V, gamma=1):
    policy = np.zeros([env.observation_space.n, env.action_space.n]) / env.action_space.n
    for s in range(env.observation_space.n):
        q = q_from_v_soln(env, V, s, gamma)
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.action_space.n)[i] for i in best_a], axis=0)/len(best_a)
    return policy

def policy_iteration_soln(env, gamma=1, theta=1e-8):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation_soln(env, policy, gamma, theta)
        new_policy = policy_improvement_soln(env, V)
        if (new_policy == policy).all():
            break;
        policy = copy.copy(new_policy)
    return policy, V

env = gym.make('FrozenLake-v1',
               desc=None,
               map_name='4x4',
               is_slippery=True,
               success_rate=1.0/3.0,
               reward_schedule=(1,0,0))
env.action_space.n
random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
plot = False

class Tests(unittest.TestCase):

    def policy_evaluation_check(self, policy_evaluation, plot):
        soln = policy_evaluation_soln(env, random_policy)
        to_check = policy_evaluation(env, random_policy, plot)
        np.testing.assert_array_almost_equal(soln, to_check)

    def q_from_v_check(self, q_from_v, plot):
        V = policy_evaluation_soln(env, random_policy)
        soln = np.zeros([env.observation_space.n, env.action_space.n])
        to_check = np.zeros([env.observation_space.n, env.action_space.n])
        for s in range(env.observation_space.n):
            soln[s] = q_from_v_soln(env, V, s)
            to_check[s] = q_from_v(env, V, s)
        np.testing.assert_array_almost_equal(soln, to_check)

    def policy_improvement_check(self, policy_improvement, plot):
        V = policy_evaluation_soln(env, random_policy)
        new_policy = policy_improvement(env, V)
        new_V = policy_evaluation_soln(env, new_policy)
        self.assertTrue(np.all(new_V >= V))

    def policy_iteration_check(self, policy_iteration, plot):
        policy_soln, _ = policy_iteration_soln(env)
        policy_to_check, _ = policy_iteration(env, plot)
        soln = policy_evaluation_soln(env, policy_soln)
        to_check = policy_evaluation_soln(env, policy_to_check)
        np.testing.assert_array_almost_equal(soln, to_check)

    def truncated_policy_iteration_check(self, truncated_policy_iteration, plot):
        self.policy_iteration_check(truncated_policy_iteration, False)

    def value_iteration_check(self, value_iteration, plot):
        self.policy_iteration_check(value_iteration, False)

check = Tests()

def run_check(check_name, func):
    try:
        getattr(check, check_name)(func, plot)
    except check.failureException as e:
        printmd('**<span style="color: red;">PLEASE TRY AGAIN</span>**')
        return
    printmd('**<span style="color: green;">PASSED</span>**')