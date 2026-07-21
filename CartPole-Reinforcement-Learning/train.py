import os

import gymnasium as gym
from stable_baselines3 import PPO

# Create folders if they don't exist
os.makedirs("models", exist_ok=True)

# Create CartPole environment
env = gym.make("CartPole-v1")

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

print("Training started...\n")

# Train the model
TIMESTEPS = 50000

model.learn(total_timesteps=TIMESTEPS)
model.save(f"models/cartpole_model_{TIMESTEPS}")

# Save the trained model
model.save("models/cartpole_model")

print("\nTraining completed successfully!")
print("Model saved inside models/cartpole_model.zip")

env.close()