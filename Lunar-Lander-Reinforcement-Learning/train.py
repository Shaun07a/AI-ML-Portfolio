import os

import gymnasium as gym
from stable_baselines3 import PPO

# Create models directory
os.makedirs("models", exist_ok=True)

# Create Lunar Lander environment
env = gym.make("LunarLander-v3")

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

TIMESTEPS = 500000

print("Training Lunar Lander Agent...\n")

# Train the model
model.learn(total_timesteps=TIMESTEPS)

# Save the model
model.save("models/lunar_lander_model")

print("\nTraining Complete!")
print("Model saved to models/lunar_lander_model.zip")

env.close()