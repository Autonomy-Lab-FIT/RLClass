import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()