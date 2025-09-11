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

"""
TERMINAL COMMANDS:

Training + Evaluation --
python meta_teacher_student.py --task pushing --student_train_steps 50000 --meta_episodes 50 --eval --seed 0 --log_dir logs/autocalc_logs --use_wandb

Short Training (for testing) --
python meta_teacher_student.py --task pushing --student_train_steps 5000 --meta_episodes 5 --eval --seed 0 --log_dir logs/autocalc_test --use_wandb
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import REACHING_BENCHMARK, PUSHING_BENCHMARK, PICKING_BENCHMARK, PICK_AND_PLACE_BENCHMARK, STACKING2_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy,
    VisualInterventionActorPolicy,
    RigidPoseInterventionActorPolicy,
    RandomInterventionActorPolicy
)
from causal_world.task_generators import generate_task

# importing the resuable components from baselines and validation_actor
# because why not
from baselines import (
    INTERVENTIONS,
    TASK_BENCHMARKS,
    SUPPORTED_TASKS,
    DENSE_REWARD_WEIGHTS,
    create_environment,
    evaluate_cm_score,
    test_intervention_performance,
    RewardMonitorCallback,
    IntervenedCausalWorld,
    CSVLogger,
    ValidationCallback
)
from validation_actor import ValidationInterventionActorPolicy
import wandb
from wandb.integration.sb3 import WandbCallback

# obtaining the exploration rate
def get_exploration_rate(meta_step, meta_episodes):
    """
    advanced exploration strategy with three phases:
    1. initial pure exploration phase
    2. guided exploration phase with faster decay
    3. fine-tuning phase with minimal exploration
    """
    # phase 1: initial pure exploration (first 10% of episodes)
    if meta_step < meta_episodes * 0.1:
        return 1.0

    # phase 2: guided exploration (next 60% of episodes)
    elif meta_step < meta_episodes * 0.7:
        # start at 0.8 and decay to 0.2 over this phase
        phase_progress = (meta_step - meta_episodes * 0.1) / (meta_episodes * 0.6)
        return 0.8 - 0.6 * phase_progress
    
    # phase 3: fine-tuning (last 30% of episodes)
    else:
        # low exploration rate for fine-tuning
        return 0.2 - 0.15 * ((meta_step - meta_episodes * 0.7) / (meta_episodes * 0.3))

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
    Compute the meta-state for the teacher by evaluating reward and CM score for each intervention
    Returns interleaved array: [reward1, cm1, reward2, cm2, ..., reward7, cm7]
    """
    logging.info("Computing teacher state (rewards and CM scores for all interventions)")
    cm_scores = []
    rewards = []

    for i, intervention in enumerate(interventions):
        logging.info(f"     Processing intervention {i+1}/{len(interventions)}: {intervention['type']}")

        # 1. Test the reward performance
        reward_metrics = test_intervention_performance(
            student_model,
            intervention,
            task_name,
            num_episodes=5,
            seed=seed + i
        )
        rewards.append(reward_metrics['avg_reward'])
        logging.info(f"     {intervention['type']} reward: {reward_metrics['avg_reward']:.4f}")

        # 2. Compute CM score
        env = create_environment(task_name, intervention, seed=seed + i + 100)
        cm_score = evaluate_cm_score(
            env, 
            student_model, 
            max_episodes=5, 
            device=device,
            intervention_type=intervention['type'], 
            seed=seed + i + 100
        )
        
        cm_scores.append(cm_score)
        env.close()
        logging.info(f"     {intervention['type']} CM score: {cm_score:.4f}")
    
    # normalize reward scores
    reward_array = np.array(rewards, dtype=np.float32)
    reward_mean = np.mean(reward_array)
    reward_std = np.std(reward_array)
    if reward_std > 1e-8:
        normalized_rewards = (reward_array - reward_mean) / reward_std
    else:
        normalized_rewards = reward_array - reward_mean
    
    # normalize CM scores
    cm_array = np.array(cm_scores, dtype=np.float32)  # Fixed: was using rewards instead of cm_scores
    cm_mean = np.mean(cm_array)
    cm_std = np.std(cm_array)
    if cm_std > 1e-8:
        normalized_cm = (cm_array - cm_mean) / cm_std
    else:
        normalized_cm = cm_array - cm_mean
    
    # interleave rewards and CM scores
    interleaved_state = np.zeros(len(interventions) * 2, dtype=np.float32)
    interleaved_state[0::2] = normalized_rewards    # even indices for rewards
    interleaved_state[1::2] = normalized_cm         # odd indices for CM scores
    
    logging.info(f"Raw rewards: {reward_array}")
    logging.info(f"Raw CM scores: {cm_array}")
    logging.info(f"Normalized rewards: {normalized_rewards}")
    logging.info(f"Normalized CM scores: {normalized_cm}")
    logging.info(f"Interleaved state shape: {interleaved_state.shape}")
    
    return interleaved_state

# evaluate student generalization
def run_validation_protocol(student_model, task_name, stage_num, args, csv_logger, cumulative_timesteps):
    """Evaluate generalization performance on the 10, fixed validation environments"""
    logging.info(f"[VALIDATION] Running validation after meta-episode {stage_num}")

    # create validation callback with the same settings as baselines
    validation_callback = ValidationCallback(
        validation_frequency = float('inf'),    # only run when explicitly called
        task_name=task_name,
        csv_logger=csv_logger,
        stage=stage_num,
        cumulative_timesteps=cumulative_timesteps,
        validation_episodes=10,
        seed=args.seed + stage_num * 10000,
        baseline_type="autocalc"
    )
    
    # run validation and return metrics
    validation_metrics = validation_callback._execute_validation(student_model)

    return validation_metrics

def test_all_interventions_for_teacher(student_model, task_name, stage_num, args, csv_logger, cumulative_timesteps, selected_intervention_type):
    """Test student model on all interventions to gather data for teacher state and logging"""
    logging.info(f"[AUTOCALC] Testing all interventions at meta-episode {stage_num}")
    
    # Test each intervention type including none
    for intervention in INTERVENTIONS + [None]:
        int_type = intervention['type'] if intervention is not None else 'none'
        
        # Test performance on this intervention
        metrics = test_intervention_performance(
            student_model=student_model,
            intervention=intervention,
            task_name=task_name,
            num_episodes=getattr(args, 'test_episodes', 10),
            seed=args.seed + stage_num * 100 + (INTERVENTIONS.index(intervention) if intervention in INTERVENTIONS else 0)
        )
        
        # Calculate CM score if needed for teacher state
        env = create_environment(task_name, intervention, seed=args.seed + stage_num * 100)
        cm_score = evaluate_cm_score(
            env, 
            student_model, 
            max_episodes=5, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            intervention_type=int_type, 
            seed=args.seed + stage_num * 100
        )
        env.close()
        
        # Log results to CSV
        if csv_logger:
            csv_logger.log_intervention_test(
                stage=stage_num,
                intervention_type=int_type,
                test_avg_reward=metrics['avg_reward'],
                test_success_rate=metrics['success_rate'],
                test_avg_length=metrics['avg_length'],
                selected=(int_type == selected_intervention_type),  # Mark which intervention was selected
                cumulative_timesteps=cumulative_timesteps,
                cm_score=cm_score,
                baseline_type="autocalc"
            )

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

# TODO: modify this to leakyrelu if you get a vanishing gradient issue
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
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu', buffer_size=50000, batch_size=64, target_update=5):
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

        # self.update_count += 1
        # if self.update_count % self.target_update == 0:
        #     self.target_net.load_state_dict(self.q_net.state_dict())
        
        # replacing hard update with:
        tau = 0.005
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return loss.item()

# the eval function
def evaluate_autocalc_performance(log_dir, task_name, eval_model, seed=0, max_episode_length=250, skip_frame=3, num_episodes=10):
    """made similar to baselines evaluation"""
    set_seed(seed)
    logging.info("Running AutoCaLC evaluation...")

    # create env
    dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
    task = generate_task(
        task_generator_id=task_name,
        dense_reward_weights=np.array(dense_weights),
        variables_space='space_a',
        fractional_reward_weight=1
    )
    env = CausalWorld(
        task=task,
        skip_frame=skip_frame,
        action_mode='joint_torques',
        enable_visualization=False,
        seed=seed,
        max_episode_length=max_episode_length
    )

    # load the final model
    model_path = os.path.join(log_dir, eval_model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    model = PPO.load(model_path)

    # basic episode evaluation
    logging.info("\nFirst phase of evaluation:")
    all_rewards, all_successes = [], []
    for ep in range(num_episodes):
        obs = env.reset()
        if hasattr(env, 'seed'):
            env.seed(seed + ep)
        done = False
        total_reward, successes = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if isinstance(info, dict) and 'success' in info:
                successes += int(info['success'])
        logging.info(f"Episode {ep + 1}: reward = {total_reward:.2f}, success = {successes}")
        all_rewards.append(total_reward)
        all_successes.append(successes)
    
    logging.info(f"\nMean reward: {np.mean(all_rewards):.2f}")
    logging.info(f"Mean success rate: {np.mean(all_successes):.2f}")

    # benchmark eval (same as baselines)
    logging.info("\nGenerating benchmark evaluation and viz...")
    if task_name not in TASK_BENCHMARKS:
        logging.error(f"No benchmark available for task: '{task_name}'. Supported: {SUPPORTED_TASKS}")
        return
    benchmark = TASK_BENCHMARKS[task_name]

    # use the causalworld benchmark evaluation
    evaluation = EvaluationPipeline(
        evaluation_protocols=benchmark['evaluation_protocols'],
        task_params={'task_generator_id': task_name},
        world_params={'skip_frame': 3, 'action_mode': 'joint_torques'},
        visualize_evaluation=False
    )

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    scores_model = evaluation.evaluate_policy(policy_fn, fraction=0.005)

    logging.info("\nEvaluation results:")
    logging.info(scores_model)

    # save the benchmark results
    import json
    benchmark_path = os.path.join(log_dir, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump({
            'final_evals': scores_model
        }, f, indent=2)
    logging.info(f"Final evals saved to: {benchmark_path}")

    # generate the visualizations
    plots_dir = os.path.join(log_dir, "plots")
    vis.generate_visual_analysis(plots_dir, experiments={task_name: scores_model})
    logging.info(f"Visualization saved to: {plots_dir}")

# ========================
# main meta-learning loop
# ========================
def main():
    # Get arguments from the global parser (already parsed in __main__)
    import sys
    # Re-parse arguments in main for clarity
    parser = argparse.ArgumentParser(description="Meta-RL Teacher-Student Curriculum (AutoCaLC)")
    parser.add_argument('--task', type=str, default='pushing', help='CausalWorld task name')
    parser.add_argument('--student_train_steps', type=int, default=50000, help='Timesteps per student training block')
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes (teacher steps)')
    parser.add_argument('--student_pretrained_path', type=str, default=None, help='Path to pretrained PPO model (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_dir', type=str, default='logs/autocalc', help='Log directory')
    parser.add_argument('--test_episodes', type=int, default=10, help='Episodes for testing each intervention')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation environment')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip for environment')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    args = parser.parse_args()

    # the initial setup phase
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.log_dir, 'autocalc.log')), logging.StreamHandler()])
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize csv logger
    csv_logger = CSVLogger(args.log_dir)

    if args.student_pretrained_path is None:
        args.student_pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'
        logging.info(f"Auto-determined pretrained path: {args.student_pretrained_path}")
    
    if args.use_wandb:
        wandb.init(
            project=f'autocalc-{args.task}',
            name=f'autocalc_{args.task}_seed{args.seed}',
            config=vars(args),
            tags=[args.task, 'autocalc', 'meta-learning']
        )
    
    logging.info(f"Starting AutoCaLC training for task '{args.task}' on device '{device}'")

    # --- 1. Initialization ---
    student_model = PPO.load(args.student_pretrained_path, device=device)
    logging.info(f"Loaded student PPO from {args.student_pretrained_path}")

    teacher = TeacherDQNAgent(state_dim=len(INTERVENTIONS) * 2, action_dim=len(INTERVENTIONS), device=device)

    # Track cumulative timesteps
    cumulative_timesteps = 0
    
    # Get initial validation performance
    initial_validation_metrics = run_validation_protocol(
        student_model, args.task, 0, args, csv_logger, cumulative_timesteps
    )
    last_validation_metrics = initial_validation_metrics

    # --- 2. Main meta-learning loop ---
    for meta_step in range(args.meta_episodes):
        logging.info(f"\n{'='*20} Meta-Episode {meta_step + 1}/{args.meta_episodes} {'='*20}")

        # 3. Compute current meta-state
        current_meta_state = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed + meta_step)

        # 4. Teacher selects an intervention
        epsilon = get_exploration_rate(meta_step, args.meta_episodes)
        selected_intervention_idx = teacher.select_action(current_meta_state, epsilon)
        selected_intervention = INTERVENTIONS[selected_intervention_idx]
        selected_intervention_type = selected_intervention['type']
        logging.info(f"Teacher chose intervention: '{selected_intervention_type}' (Epsilon: {epsilon:.2f})")

        # 5. Student trains on that intervention
        train_env = create_environment(args.task, selected_intervention, seed=args.seed + meta_step * 100)
        vec_env = DummyVecEnv([lambda: train_env])
        vec_env = VecMonitor(vec_env, filename=os.path.join(args.log_dir, f'autocalc_monitor_stage{meta_step+1}.csv'))
        
        # Set up training monitoring
        reward_monitor = RewardMonitorCallback(
            selected_intervention_type, csv_logger, meta_step+1, cumulative_timesteps, baseline_type="autocalc"
        )
        
        # Create callback list
        callback_list = CallbackList([
            reward_monitor,
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=args.log_dir if args.use_wandb else None,
                verbose=2
            ) if args.use_wandb else None
        ])
        callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]
        
        # Train the student
        student_model.set_env(vec_env)
        student_model.learn(
            total_timesteps=args.student_train_steps,
            callback=callback_list,
            reset_num_timesteps=False
        )
        
        # Update cumulative timesteps
        cumulative_timesteps += args.student_train_steps
        
        # 6. Test on all interventions for logging
        test_all_interventions_for_teacher(
            student_model, args.task, meta_step+1, args, csv_logger, cumulative_timesteps, selected_intervention_type
        )
        
        # 7. Evaluate new generalization performance
        current_validation_metrics = run_validation_protocol(
            student_model, args.task, meta_step+1, args, csv_logger, cumulative_timesteps
        )

        # 8. Calculate the meta-reward using the new formula
        meta_reward = (10 * current_validation_metrics['validation_success_rate']) + current_validation_metrics['validation_avg_reward']
        logging.info(f"Meta-Reward: {meta_reward:.4f} (Success Rate: {current_validation_metrics['validation_success_rate']:.3f}, Avg Reward: {current_validation_metrics['validation_avg_reward']:.3f})")

        # 9. Compute the next meta-state
        next_meta_state = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed + meta_step + 1)

        # 10. Store experience and update the teacher
        teacher.replay_buffer.push(current_meta_state, selected_intervention_idx, meta_reward, next_meta_state)
        teacher_loss = teacher.train_step()
        logging.info(f"Teacher training step complete. Loss: {teacher_loss if teacher_loss is not None else 'N/A'}")

        # 11. Log to wandb
        if args.use_wandb:
            wandb.log({
                'meta_episode': meta_step + 1,
                'teacher_action': selected_intervention_type,
                'teacher_reward': meta_reward,
                'student_validation_success_rate': current_validation_metrics['validation_success_rate'],
                'student_validation_reward': current_validation_metrics['validation_avg_reward'],
                'teacher_epsilon': epsilon,
                'teacher_loss': teacher_loss,
                'cumulative_timesteps': cumulative_timesteps,
                **{f"cm_score_{inter['type']}": score for inter, score in zip(INTERVENTIONS, current_meta_state[1::2])}
            })
        
        stage_model_path = os.path.join(args.log_dir, f"student_model_stage_{meta_step+1}.zip")
        student_model.save(stage_model_path)
        logging.info(f"Saved intermediate model for stage {meta_step+1} to {stage_model_path}")
        
        # Clean up
        train_env.close()
    
    # --- And now the final step ---
    final_model_path = os.path.join(args.log_dir, "final_student_model.zip")
    student_model.save(final_model_path)
    logging.info(f"Meta-RL training complete! Final student model saved to {final_model_path}")
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta-RL Teacher-Student Curriculum (AutoCaLC)")
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--task', type=str, default='pushing', help='CausalWorld task name')
    parser.add_argument('--student_train_steps', type=int, default=50000, help='Timesteps per student training block')
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes (teacher steps)')
    parser.add_argument('--student_pretrained_path', type=str, default=None, help='Path to pretrained PPO model (optional)')
    parser.add_argument('--pretrained_eval', type=str, help='Path to eval model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory (will be auto-generated if not specified)')
    parser.add_argument('--test_episodes', type=int, default=10, help='Episodes for testing each intervention')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation environment')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip for environment')
    args = parser.parse_args()

    # Set default log directory if not specified
    if args.log_dir is None:
        if args.eval:
            raise ValueError("--log_dir must be specified for evaluation")
        else:
            args.log_dir = f'logs/autocalc_{args.task}_seed{args.seed}'

    if args.eval:
        evaluate_autocalc_performance(args.log_dir, task_name=args.task, eval_model=args.pretrained_eval, seed=args.seed)
    else:
        main()
