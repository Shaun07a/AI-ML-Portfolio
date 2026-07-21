# 🚀 Lunar Lander Reinforcement Learning Agent

A Reinforcement Learning project that trains an intelligent agent to safely land a spacecraft using the **Proximal Policy Optimization (PPO)** algorithm from **Stable-Baselines3**. The agent interacts with the **Gymnasium LunarLander-v3** environment and learns optimal landing strategies through trial and error.

---

## 📌 Features

- Reinforcement Learning using PPO
- Gymnasium LunarLander-v3 environment
- Model training and evaluation
- Save and load trained models
- Easy-to-understand project structure
- Beginner-friendly implementation

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
Lunar-Lander-Reinforcement-Learning/
│
├── models/
│   └── lunar_lander_model.zip
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
git clone https://github.com/Shaun07a/Lunar-Lander-Reinforcement-Learning.git
```

### Move into the project directory

```bash
cd Lunar-Lander-Reinforcement-Learning
```

### Create a virtual environment

```bash
python -m venv lunar
```

### Activate the virtual environment

**Windows**

```bash
lunar\Scripts\activate
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

A simulation window will open where the trained agent attempts to land the spacecraft.

---

## 🧠 Algorithm

This project uses the **Proximal Policy Optimization (PPO)** algorithm.

### Training Workflow

1. Create the LunarLander environment.
2. Initialize the PPO agent.
3. Train the agent by interacting with the environment.
4. Save the trained model.
5. Load the model for evaluation.
6. Test the landing performance.

---

## 🎯 Objective

The objective is to land the spacecraft safely on the designated landing pad while minimizing unnecessary movement and avoiding crashes.

---

## 📊 Environment

**Environment Name**

```
LunarLander-v3
```

### Observation Space

The agent observes:

- Position
- Velocity
- Angle
- Angular Velocity
- Left Leg Contact
- Right Leg Contact

### Action Space

| Action | Description |
|---------|-------------|
| 0 | Do Nothing |
| 1 | Fire Left Engine |
| 2 | Fire Main Engine |
| 3 | Fire Right Engine |

---

## 📈 Future Improvements

- DQN implementation
- A2C implementation
- SAC implementation
- Hyperparameter tuning
- TensorBoard integration
- Training performance graphs
- Automatic video recording
- Web dashboard for model evaluation

---

## 👨‍💻 Author

**Shaun Joseph**


