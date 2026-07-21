# 🤖 CartPole Reinforcement Learning Agent

A Reinforcement Learning project that trains an intelligent agent to balance a pole on a moving cart using the **Proximal Policy Optimization (PPO)** algorithm from **Stable-Baselines3**. The environment is provided by **Gymnasium**, and the trained model learns through trial and error to maximize the episode reward.

---

## 📌 Features

- Reinforcement Learning using PPO
- OpenAI Gymnasium CartPole-v1 environment
- Model training and evaluation
- Save and load trained models
- Easy-to-use Python scripts
- Beginner-friendly project structure

---

## 🛠️ Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- Pygame

---

## 📂 Project Structure

```
CartPole-Reinforcement-Learning/
│
├── models/
│   └── cartpole_model.zip
│
├── train.py
├── test.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Shaun07a/CartPole-Reinforcement-Learning.git
```

### Move into the project directory

```bash
cd CartPole-Reinforcement-Learning
```

### Create a virtual environment

```bash
python -m venv cartpole
```

### Activate the virtual environment

**Windows**

```bash
cartpole\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Training the Agent

Run:

```bash
python train.py
```

The trained model will be saved in the **models/** directory.

---

## 🎮 Testing the Agent

Run:

```bash
python test.py
```

A simulation window will open where the trained agent balances the pole automatically.

---

## 🧠 Algorithm

This project uses the **Proximal Policy Optimization (PPO)** algorithm.

### Training Workflow

1. Create the CartPole environment.
2. Initialize the PPO agent.
3. Train the agent through interaction with the environment.
4. Save the trained model.
5. Load the saved model for testing.
6. Evaluate the agent in the simulation.

---

## 🎯 Objective

The goal is to keep the pole balanced by moving the cart left or right.

The episode ends when:

- The pole falls beyond the allowed angle.
- The cart moves outside the allowed boundary.
- The maximum episode length is reached.

---

## 📊 Environment

**Environment Name**

```
CartPole-v1
```

### Observation Space

The agent observes:

- Cart Position
- Cart Velocity
- Pole Angle
- Pole Angular Velocity

### Action Space

| Action | Description |
|---------|-------------|
| 0 | Move Left |
| 1 | Move Right |

---

## 📈 Future Improvements

- DQN implementation
- A2C implementation
- Hyperparameter tuning
- TensorBoard integration
- Training performance graphs
- Video recording of trained agent
- Streamlit dashboard for model evaluation

---

## 👨‍💻 Author

**Shaun Joseph**



