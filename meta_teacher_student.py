"""
PSEUDOCODE
# 1. initialization
student_agent = PPO(…)
teacher_agent = DQN(state_dim=14, action_dim=7, …)
validation_env = create_validation_env()

# get initial generalization performance
last_validation_performance = run_validation_protocol(student_agent, validation_env)

# 2. main meta-learning loop (e.g., for M meta-episodes)
for meta_step in range(M=50):
 	# 3. compute the current meta-state
 	current_meta_state = compute_meta_state(student_agent, INTERVENTIONS)
	
	# 4. teacher selects an intervention
	selected_intervention_idx = teacher_agent.predict(current_meta_state)
	selected_intervention = INTERVENTIONS[selected_intervention_idx]

 	# 5. student trains on that intervention
 	student_agent.learn(total_timesteps=50000, env=create_intervened_env(selected_intervention))

	# 6. evaluate new generalization performance
	current_validation_performance = run_validation_protocol(student_agent, validation_env)

	# 7. calculate the meta-reward
	meta_reward = current_validation_performance - last_validation_performance
	last_validation_performance = current_validation_performance

	# 8. compute the next meta-state
	next_meta_state = compute_meta_state(student_agent, INTERVENTIONS)
	
	# 9. store experience and update the teacher
 	teacher_agent.replay_buffer.add(current_meta_state, selected_intervention_idx, meta_reward, next_meta_state, done)
	teacher_agent.train(batch_size=…)

    
HELPER FUNCTIONS
1. compute_meta_state (student, interventions)
    - this function iterates through all available interventions
    - for each intervention, we call the existing test_intervention_performance to probe the reward and evaluate_cm_score
    to get the novelty score
    - we then concatenate all these values into a 14-dimensional state vector and return that

2. run_validation_protocol (student, validation_env)
    - this function is responsible for measuring generalization
    - it takes the current student and evaluates it for 50 episodes on the fixed validation env
    - it should return a single, robust performance metric, such as the average performance metric, which will be used to
    calculate the meta-reward

NOTES (tuning the meta-learning process)
- setting 50K timesteps for each student training block; this gives the student enough time to adapt to the intervention
- the teacher's exploration (epsilon) may need to decay slowly since each data-point (meta-step) is expensive to collect
and we have 50 meta-episodes for a full training run; must tune epsilon based on the meta-step we're on, not the timestep
- we would use a large replay buffer for the teacher's DQN to help mitigate the non-stationary of the agent
- we use a slow update frequency (low tau) for the DQN's target network to improve stability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import argparse
import logging
from collections import deque
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

# importing the resuable components from baselines and validation_actor
# because why not
from baselines import (
    INTERVENTIONS,
    DENSE_REWARD_WEIGHTS,
    create_environment,
    evaluate_cm_score,
    RewardMonitorCallback,
    IntervenedCausalWorld
)
from validation_actor import ValidationInterventionActorPolicy
import wandb
from wandb.integration.sb3 import WandbCallback


# setting random seed
def set_seed(seed):
    """set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"random seed set to {seed}")

# get the teacher state
def get_teacher_state(student_model, task_name, interventions, device='cpu', seed=0):
    """
    this func computes the meta-state for the teacher by evaluating the CM score for the
    student agent under each available intervention
    TODO: I also want the reward
    so if we have 7 interventions, it would be:
    [reward1, cm1, reward2, cm2, ..., reward7, cm7]
    """
    logging.info("computing teacher state (cm scores for all interventions - for now)")
    cm_scores = []
    rewards = []
    for i, intervention in enumerate(interventions):
        logging.info(f"     Processing intervention {i+1}/{len(interventions)}: {intervention['type']}")
        # use the create_env factory from baselines.py
        env = create_environment(task_name, intervention, seed=seed)
        # use the evaluate_cm_score func from baselines.py
        cm_score = evaluate_cm_score(env, student_model, episodes=5, device=device, intervention_type=intervention['type'], seed=seed)
        cm_scores.append(cm_score)
        env.close()
        logging.info(f"     {intervention['type']} CM score: {cm_score:.4f}")
    
    cm_array = np.array(cm_scores, dtype=np.float32)

    # normalize scores for stable dqn training
    mean = np.mean(cm_array)
    std = np.std(cm_array)
    if std > 1e-8:
        normalized_cm = (cm_array - mean) / std
    else:
        normalized_cm = cm_array - mean    # here we center if std is zero
    
    logging.info(f"Raw CM scores: {cm_array}")
    logging.info(f"Normalized CM scores: {normalized_cm}")
    return normalized_cm

# evaluate student generalization
def run_validation_protocol(student_model, validation_env, num_episodes=50):
    """
    evaluate generalization performance on a fixed validation env
    primary metric: the success rate, we can use this as the meta-reward
    """
    logging.info(f"Running validation protocol ({num_episodes} episodes)...")
    successes = 0
    total_rewards = 0

    for _ in range(num_episodes):
        obs = validation_env.reset()
        done = False
        episode_success = False
        while not done:
            act, _ = student_model.predict(obs, deterministic=False)
            obs, reward, done, info = validation_env.step(act)
            total_rewards += reward
            if isinstance(info, dict) and info.get('success') and not episode_success:
                successes += 1
                episode_success = True
                done = True     # early stop on success
    
    success_rate = successes / num_episodes
    avg_reward = total_rewards / num_episodes

    logging.info(f"Validation Performance: Success Rate={success_rate:.3f}, Avg Reward={avg_reward:.3f}")
    return success_rate

# the teacher dqn agent
class ReplayBuffer:
    def __init__(self, capacity, state_dim, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s, a, r, s_):
        self.buffer.append((s, a, r, s_))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_ = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32).to(self.device),
            torch.tensor(a, dtype=torch.long).to(self.device),
            torch.tensor(r, dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(s_), dtype=torch.float32).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TeacherDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu', buffer_size=100, batch_size=8, target_update=5):
        self.device = device
        self.q_net = DQN(state_dim, action_dim).to(device)
        self.target_net = deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, device)
        self.update_count = 0
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_net.net[-1].out_features - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(q_values.argmax().item())
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        s, a, r, s_ = self.replay_buffer.sample(self.batch_size)
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(s_).max(1)[0]
            target = r + self.gamma * max_next_q
        
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

# ========================
# main meta-learning loop
# ========================
def main():
    parser = argparse.ArgumentParser(description="Meta-RL Teacher-Student Curriculum (AutoCaLC)")
    parser.add_argument('--task', type=str, default='pushing', help='CausalWorld task name')
    parser.add_argument('--student_train_steps', type=int, default=50000, help='Timesteps per student training block')
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes (teacher steps)')
    # TODO: this should not be optional btw, should load automatically as it did for the baselines
    # oh no i get it now
    parser.add_argument('--student_pretrained_path', type=str, default=None, help='Path to pretrained PPO model (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_dir', type=str, default='autocalc_logs', help='Log directory')
    args = parser.parse_args()

    # the initial setup phase
    os.make_dirs(args.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.log_dir, 'autocalc.log')), logging.StreamHandler()])
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.student_pretrained_path is None:
        args.student_pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'
        logging.info(f"Auto-determined pretrained path: {args.student_pretrained_path}")
    
    if args.use_wandb:
        wandb.init(project=f'autocalc-{args.task}', name=f'autocalc_{args.task}_seed{args.seed}', config=vars(args))
    
    logging.info(f"Starting AutoCaLC training for task '{args.task}' on device '{device}'")

    # --- 1. Initialization ---
    student_model = PPO.load(args.student_pretrained_path, device=device)
    logging.info(f"Loaded student PPO from {args.student_pretrained_path}")

    teacher = TeacherDQNAgent(state_dim=len(INTERVENTIONS), action_dim=len(INTERVENTIONS), device=device)

    # create a validation env with an intervention set
    validation_intervention = {"type": "validation", "class": ValidationInterventionActorPolicy, "params": {"seed": args.seed + 42}}
    validation_env = create_environment(args.task, validation_intervention, seed=args.seed + 42)

    last_validation_performance = run_validation_protocol(student_model, validation_env)
    logging.info(f"Initial student validation performance (success rate): {last_validation_performance:.3f}")

    # --- 2. Main meta-learning loop ---
    for meta_step in range(args.meta_episodes):
        logging.info(f"\n{'='*20} Meta-Episode {meta_step + 1}/{args.meta_episodes} {'='*20}")

        # 3. Compute current meta-state
        current_meta_state = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed + meta_step)

        # 4. Teacher selects an intervention
        epsilon = max(0.1, 1.0 - meta_step / (args.meta_episodes * 0.8))    # epsilon decay
        selected_intervention_idx = teacher.select_action(current_meta_state, epsilon)
        selected_intervention = INTERVENTIONS[selected_intervention_idx]
        logging.info(f"Teacher chose intervention: '{selected_intervention['type']}' (Epsilon: {epsilon:.2f})")

        # 5. Student trains on that intervention
        train_env = create_environment(args.task, selected_intervention, seed=args.seed + meta_step * 100)
        vec_env = DummyVecEnv([lambda: train_env])
        vec_env = VecMonitor(vec_env)
        student_model.set_env(vec_env)
        student_model.learn(total_timesteps=args.student_train_steps, reset_num_timesteps=False)
        train_env.close()

        # 6. Evaluate the new generealization performance for meta-reward
        current_validation_performance = run_validation_protocol(student_model, validation_env)

        # 7. Calculate the meta-reward
        meta_reward = current_validation_performance - last_validation_performance
        last_validation_performance = current_validation_performance
        logging.info(f"Meta-Reward: {meta_reward:.4f} (New Perf: {current_validation_performance:.3f})")

        # 8. Compute the next meta-state
        next_meta_state = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed + meta_step + 1)

        # 9. Store experience and update teacher
        teacher.replay_buffer.push(current_meta_state, selected_intervention_idx, meta_reward, next_meta_state)
        teacher_loss = teacher.train_step()
        logging.info(f"Teacher training step complete. Loss: {teacher_loss if teacher_loss is not None else 'N/A'}")

        # time to log!!
        if args.use_wandb:
            wandb.log({
                'meta_episode': meta_step + 1,
                'teacher_action': selected_intervention['type'],
                'teacher_reward': meta_reward,
                'student_validation_success_rate': current_validation_performance,
                'teacher_epsilon': epsilon,
                'teacher_loss': teacher_loss,
                **{f"cm_score_{inter['type']}": score for inter, score in zip(INTERVENTIONS, current_meta_state)}
            })
    
    # --- And now the final step ---
    final_model_path = os.path.join(args.log_dir, "final_student_model.zip")
    student_model.save(final_model_path)
    logging.info(f"Meta-RL training complete! Final student model saved to {final_model_path}")
    validation_env.close()
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
