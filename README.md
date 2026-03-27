# EduCode Rwanda - Reinforcement Learning Tutor Agent

This project implements a reinforcement learning environment that simulates
an AI tutor navigating a student's learning journey through Rwanda's TVET
Level 3 JavaScript curriculum. The agent decides what instructional action
to take at each step to maximise student mastery while keeping engagement high.

The environment is built on top of the Gymnasium interface. Four RL algorithms
are trained and compared: DQN, REINFORCE, PPO, and A2C, all using the same
custom environment for a fair comparison.

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py       custom Gymnasium environment
│   └── rendering.py        pygame visualisation
├── training/
│   ├── dqn_training.py     DQN training with 10 hyperparameter runs
│   └── pg_training.py      REINFORCE, PPO, A2C training (10 runs each)
├── models/
│   ├── dqn/                saved DQN models
│   └── pg/                 saved policy gradient models
├── main.py                 entry point for running the best agent
├── requirements.txt        project dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or higher is recommended.

## Running the Environment Demo (Random Agent)

This shows the visualisation without any training involved:

```bash
python main.py --mode random --episodes 3 --profile average
```

## Training the Agents

Train DQN (all 10 hyperparameter runs):
```bash
python training/dqn_training.py
```

Train policy gradient methods:
```bash
python training/pg_training.py --algo all
```

Run a specific algorithm:
```bash
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
python training/pg_training.py --algo reinforce
```

## Running the Best Trained Agent

```bash
python main.py --mode trained --algo ppo --episodes 5 --profile average
```

For a struggling student profile:
```bash
python main.py --mode trained --algo ppo --profile struggling
```

Without the GUI (terminal only):
```bash
python main.py --mode trained --algo ppo --no_render
```

## Environment Details

The agent (AI tutor) observes 9 continuous values describing the student's
current state and chooses from 8 discrete instructional actions.

Observation space:
- current topic index (normalised)
- mastery of current topic
- recent error rate
- engagement level
- consecutive correct answers (normalised)
- consecutive wrong answers (normalised)
- time spent on current topic
- current difficulty level
- fraction of topics completed

Actions:
- 0: explain concept
- 1: give easy exercise
- 2: give hard exercise
- 3: provide hint
- 4: give feedback
- 5: reduce difficulty
- 6: advance to next topic
- 7: run assessment

Student profiles available: struggling, average, advanced

## Mission Context

This RL solution is part of the EduCode Rwanda research project, which aims
to develop an AI-powered learning platform for JavaScript education in Rwanda's
TVET schools. The RL agent represents the adaptive tutoring component that
decides the optimal sequence of instructional interventions for each student.
