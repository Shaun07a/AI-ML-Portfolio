import gymnasium as gym
from stable_baselines3 import PPO
import time

# Create environment with rendering
env = gym.make("LunarLander-v3", render_mode="human")

# Load trained model
model = PPO.load("models/lunar_lander_model")

obs, info = env.reset()

done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    time.sleep(0.02)

env.close()

print("Episode Finished!")