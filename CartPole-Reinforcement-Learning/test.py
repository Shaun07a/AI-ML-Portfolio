import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make("CartPole-v1", render_mode="human")

model = PPO.load("models/cartpole_model")

obs, info = env.reset()

done = False

while not done:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    time.sleep(0.02)

env.close()

print("Episode Finished!")