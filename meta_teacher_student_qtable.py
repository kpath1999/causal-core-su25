"""
AutoCaLC (meta-teacher/student) runner with a tabular Q-learning teacher.

Teacher: a simple Q-table that learns optimal intervention sequences using Upper Confidence Bound (UCB) exploration.

Steps
- Iterate over M meta-episodes:
    1) Current state = last intervention used (discrete index)
    2) Teacher selects next intervention using Q-table + UCB exploration bonus
    3) Student trains K steps on selected intervention
    4) Evaluate on fixed validation environments
    5) Meta-reward = delta(validation), update Q-table using Bellman equation

Terminal Command
python meta_teacher_student_qtable.py --task pushing --meta_episodes 50 --student_train_steps 5000 --alpha 0.1 --beta 1.0 --gamma 0.9    
"""

import numpy as np
import os
import json
import argparse
import logging
import random
from collections import defaultdict
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# importing components I already created in baselines.py
from baselines import (
    INTERVENTIONS,
    TASK_BENCHMARKS,
    SUPPORTED_TASKS,
    create_environment,
    RewardMonitorCallback,
    CSVLogger,
    ValidationCallback
)

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - logging will be CSV only")

# global constants for state/action mapping
INTERVENTION_NAMES = ['goal', 'mass', 'friction', 'visual', 'position', 'angle', 'random']
INTERVENTION_TO_IDX = {
    intervention["type"]: idx
    for idx, intervention in enumerate(INTERVENTIONS)
}