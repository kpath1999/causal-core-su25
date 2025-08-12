import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import os
import logging
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy,
    VisualInterventionActorPolicy,
    RigidPoseInterventionActorPolicy,
    RandomInterventionActorPolicy
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

# === Utility: Set random seed ===
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

INTERVENTIONS = [
    {"type": "goal", "class": GoalInterventionActorPolicy, "params": {}},
    {"type": "mass", "class": PhysicalPropertiesInterventionActorPolicy, "params": {"group": "tool"}},
    {"type": "friction", "class": PhysicalPropertiesInterventionActorPolicy, "params": {"group": "stage"}},
    {"type": "visual", "class": VisualInterventionActorPolicy, "params": {}},
    {"type": "pose", "class": RigidPoseInterventionActorPolicy, "params": {"positions": True, "orientations": True}},
    {"type": "random", "class": RandomInterventionActorPolicy, "params": {}}
]

# === Models (from cp_drl.py) ===
class TransitionPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RewardPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, beta=4.0):
        super().__init__()
        self.fc_mu_logvar = nn.Linear(input_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.fc_mu_logvar(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# === CM Score Calculation Utilities ===
def train_cm_models(env, episodes=10, device='cuda'):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_dim = 64

    transition_models = [TransitionPrediction(obs_dim, act_dim, hidden_dim).to(device) for _ in range(5)]
    reward_models = [RewardPrediction(obs_dim, act_dim, hidden_dim).to(device) for _ in range(5)]
    state_models = [BetaVAE(obs_dim).to(device) for _ in range(5)]
    action_models = [BetaVAE(act_dim).to(device) for _ in range(5)]

    transition_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in transition_models]
    reward_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in reward_models]
    state_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in state_models]
    action_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in action_models]

    data = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            next_obs, rew, done, _ = env.step(act)
            data.append((obs, act, next_obs, rew))
            obs = next_obs

    states = torch.tensor([d[0] for d in data], dtype=torch.float32).to(device)
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32).to(device)
    rewards = torch.tensor([d[3] for d in data], dtype=torch.float32).to(device).unsqueeze(-1)

    for model, opt in zip(transition_models, transition_opts):
        for _ in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, next_states)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for model, opt in zip(reward_models, reward_opts):
        for _ in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()

    for model, opt in zip(state_models, state_opts):
        for _ in range(5):
            recon, mu, logvar = model(states)
            recon_loss = nn.MSELoss()(recon, states)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    for model, opt in zip(action_models, action_opts):
        for _ in range(5):
            recon, mu, logvar = model(actions)
            recon_loss = nn.MSELoss()(recon, actions)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    return transition_models, reward_models, state_models, action_models

def evaluate_cm_score(env, transition_models, reward_models, state_models, action_models, episodes=5, action_scale=1.0, device='cuda'):
    data = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            act = env.action_space.sample()
            act = act * action_scale
            next_obs, rew, done, _ = env.step(act)
            data.append((obs, act, next_obs, rew))
            obs = next_obs

    states = torch.tensor([d[0] for d in data], dtype=torch.float32).to(device)
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32).to(device)
    rewards = torch.tensor([d[3] for d in data], dtype=torch.float32).to(device).unsqueeze(-1)

    transition_std = torch.stack([m(states, actions) for m in transition_models]).std(dim=0).mean().item()
    reward_std = torch.stack([m(states, actions) for m in reward_models]).std(dim=0).mean().item()
    state_std = torch.stack([m(states)[0] for m in state_models]).std(dim=0).mean().item()
    action_std = torch.stack([m(actions)[0] for m in action_models]).std(dim=0).mean().item()

    return transition_std, reward_std, state_std, action_std

# === Dense reward weights for all tasks (from reward_curriculum_on_ppo.py) ===
DENSE_REWARD_WEIGHTS = {
    'pushing': [750, 250, 100],
    'picking': [250, 0, 125, 0, 750, 0, 0, 0.005],
    'reaching': [100000, 0, 0, 0],
    'pick_and_place': [750, 50, 250, 0, 0.005],
    'stacking2': [750, 250, 250, 125, 0.005],
}

# === Adaptive weighting function ===
def adaptive_weights(reward_scores, cm_scores, method='normalize'):
    if method == 'normalize':
        r = np.array(reward_scores)
        c = np.array(cm_scores)
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
        w_p = np.std(r_norm) / (np.std(r_norm) + np.std(c_norm) + 1e-8)
        w_cm = 1.0 - w_p
        return w_p, w_cm
    # fallback: equal weights
    return 0.5, 0.5

def make_env_with_intervention(task_name, intervention, seed=0, skip_frame=3, max_episode_length=250):
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
    # Apply intervention
    actor = intervention['class'](**intervention['params'])
    if hasattr(actor, 'initialize'):
        actor.initialize(env)
    variables_dict = env.get_variable_space_used()
    intervention_dict = actor._act(variables_dict)
    if intervention_dict:
        env.do_intervention(intervention_dict)
    return env

def evaluate_intervention(task_name, intervention, device, seed=0, cm_episodes=5, reward_episodes=5, ppo_model=None):
    env = make_env_with_intervention(task_name, intervention, seed=seed)
    # --- Compute average reward (using PPO policy if provided) ---
    total_reward = 0.0
    for _ in range(reward_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            if ppo_model is not None:
                act, _ = ppo_model.predict(obs, deterministic=True)
            else:
                act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            ep_reward += rew
        total_reward += ep_reward
    avg_reward = total_reward / reward_episodes
    # --- Compute CM score (using PPO policy if provided) ---
    data = []
    for _ in range(cm_episodes):
        obs = env.reset()
        done = False
        while not done:
            if ppo_model is not None:
                act, _ = ppo_model.predict(obs, deterministic=True)
            else:
                act = env.action_space.sample()
            next_obs, rew, done, _ = env.step(act)
            data.append((obs, act, next_obs, rew))
            obs = next_obs
    states = torch.tensor([d[0] for d in data], dtype=torch.float32).to(device)
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32).to(device)
    rewards = torch.tensor([d[3] for d in data], dtype=torch.float32).to(device).unsqueeze(-1)
    # Train CM models on this data
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_dim = 64
    transition_models = [TransitionPrediction(obs_dim, act_dim, hidden_dim).to(device) for _ in range(5)]
    reward_models = [RewardPrediction(obs_dim, act_dim, hidden_dim).to(device) for _ in range(5)]
    state_models = [BetaVAE(obs_dim).to(device) for _ in range(5)]
    action_models = [BetaVAE(act_dim).to(device) for _ in range(5)]
    transition_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in transition_models]
    reward_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in reward_models]
    state_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in state_models]
    action_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in action_models]
    for model, opt in zip(transition_models, transition_opts):
        for _ in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, next_states)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for model, opt in zip(reward_models, reward_opts):
        for _ in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for model, opt in zip(state_models, state_opts):
        for _ in range(5):
            recon, mu, logvar = model(states)
            recon_loss = nn.MSELoss()(recon, states)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    for model, opt in zip(action_models, action_opts):
        for _ in range(5):
            recon, mu, logvar = model(actions)
            recon_loss = nn.MSELoss()(recon, actions)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    t_score = torch.stack([m(states, actions) for m in transition_models]).std(dim=0).mean().item()
    r_score = torch.stack([m(states, actions) for m in reward_models]).std(dim=0).mean().item()
    s_score = torch.stack([m(states)[0] for m in state_models]).std(dim=0).mean().item()
    a_score = torch.stack([m(actions)[0] for m in action_models]).std(dim=0).mean().item()
    cm_score = t_score + r_score + s_score + a_score
    return avg_reward, cm_score, dict(transition=t_score, reward=r_score, state=s_score, action=a_score)

def score_interventions(task_name, interventions, w_p, w_cm, device, seed=0):
    results = []
    for intervention in interventions:
        logging.info(f"[AutoCaLC-CPDRL] Evaluating intervention: {intervention['type']}")
        avg_reward, cm_score, cm_details = evaluate_intervention(task_name, intervention, device, seed=seed)
        score = w_p * avg_reward + w_cm * cm_score
        results.append({
            'type': intervention['type'],
            'score': score,
            'avg_reward': avg_reward,
            'cm_score': cm_score,
            'cm_details': cm_details,
            'intervention': intervention
        })
        logging.info(f"  Reward: {avg_reward:.4f}, CM: {cm_score:.4f}, Score: {score:.4f}")
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

class CurriculumLoggingCallback(BaseCallback):
    def __init__(self, intervention_type, intervention_idx, wandb_enabled=False, verbose=0, log_mean_every=10):
        super().__init__(verbose)
        self.intervention_type = intervention_type
        self.intervention_idx = intervention_idx
        self.wandb_enabled = wandb_enabled
        self.episode_rewards = []
        self.episode_count = 0
        self.log_mean_every = log_mean_every

    def _on_step(self) -> bool:
        # Log reward and intervention at the end of each episode
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            reward = self.locals["rewards"][0]
            self.episode_rewards.append(reward)
            self.episode_count += 1
            # Print reward to terminal
            logging.info(f"[AutoCaLC-CPDRL] Episode {self.episode_count}: reward={reward:.4f} (Intervention: {self.intervention_type}, Rank: {self.intervention_idx})")
            if self.episode_count % self.log_mean_every == 0:
                mean_reward = np.mean(self.episode_rewards[-self.log_mean_every:])
                logging.info(f"[AutoCaLC-CPDRL] Mean reward (last {self.log_mean_every}): {mean_reward:.4f}")
                if self.wandb_enabled:
                    wandb.log({'train/ep_reward_mean': mean_reward}, step=self.num_timesteps)
            if self.wandb_enabled:
                wandb.log({
                    'train/episode_reward': reward,
                    'train/current_intervention': self.intervention_type,
                    'train/intervention_idx': self.intervention_idx,
                    'train/episode': self.episode_count
                }, step=self.num_timesteps)
        return True

def ppo_train_on_intervention(
    pretrained_path, env, timesteps, log_dir, seed, ppo_config, policy_kwargs, transfer_model=None, callback=None
):
    if transfer_model is not None:
        temp_path = os.path.join(log_dir, 'temp_transfer_model.zip')
        transfer_model.save(temp_path)
        model = PPO.load(temp_path, env=env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, seed=seed, **ppo_config)
        os.remove(temp_path)
    else:
        model = PPO.load(pretrained_path, env=env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, seed=seed, **ppo_config)
    model.learn(total_timesteps=timesteps, callback=callback)
    return model

# === Argparse and main entrypoint skeleton ===
def main():
    parser = argparse.ArgumentParser(description="AutoCaLC-CPDRL: Curriculum with Causal Misalignment Scoring")
    parser.add_argument('--task', type=str, default='pushing', help='Task name')
    parser.add_argument('--timesteps', type=int, default=300000, help='Timesteps per intervention')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cm_threshold', type=float, default=0.05, help='CM stabilization threshold')
    parser.add_argument('--log_dir', type=str, default='autocalc_cpdrl_logs', help='Log directory')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--cm_patience', type=int, default=2, help='Number of consecutive times CM must be below threshold to advance')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up logging to file and terminal
    log_file = os.path.join(args.log_dir, 'autocalc_cpdrl_training_debug.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(process)d %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"[AutoCaLC-CPDRL] Starting with task: {args.task}")
    logging.info(f"[AutoCaLC-CPDRL] Interventions: {[i['type'] for i in INTERVENTIONS]}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # WANDB integration
    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb.tensorboard.patch(root_logdir=args.log_dir)
        project_name = f'causal-matching-curriculum-{args.task}'
        run_name = f'causal_matching_{args.task}_seed{args.seed}'
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                'task_name': args.task,
                'curriculum_type': 'CausalMatching_Curriculum_on_PPO',
                'cm_threshold': args.cm_threshold,
                'timesteps_per_intervention': args.timesteps,
                'seed': args.seed,
            },
            tags=[args.task, 'PPO', 'CausalMatching', 'curriculum', 'adaptive'],
            sync_tensorboard=True
        )
        wandb_run = wandb.run

    # Load base PPO policy for ranking
    pretrained_path = f'ppo_{args.task}_sb3/final_model.zip'  # default naming
    if not os.path.exists(pretrained_path):
        logging.warning(f"[AutoCaLC-CPDRL] Pretrained PPO not found at {pretrained_path}. Please train a base policy first.")
        return
    base_ppo_model = PPO.load(pretrained_path)

    # Step 1: Evaluate all interventions to get CM scores using base PPO
    eval_results = []
    for intervention in INTERVENTIONS:
        logging.info(f"[AutoCaLC-CPDRL] Evaluating intervention: {intervention['type']}")
        _, cm_score, cm_details = evaluate_intervention(args.task, intervention, device, seed=args.seed, ppo_model=base_ppo_model)
        eval_results.append({
            'type': intervention['type'],
            'cm_score': cm_score,
            'cm_details': cm_details,
            'intervention': intervention
        })
        logging.info(f"  CM: {cm_score:.4f}")
        if args.use_wandb:
            wandb.log({
                f'cm_eval/{intervention["type"]}_cm_score': cm_score,
                f'cm_eval/{intervention["type"]}_cm_details': cm_details
            })

    # Step 2: Sort interventions by CM score (descending: higher mismatch prioritized)
    eval_results.sort(key=lambda x: x['cm_score'], reverse=True)

    logging.info("\n[AutoCaLC-CPDRL] Intervention ranking (by CM score):")
    for i, res in enumerate(eval_results):
        logging.info(f"  {i+1}. {res['type']} | CM: {res['cm_score']:.4f}")
    if args.use_wandb:
        wandb.log({'cm_eval/intervention_ranking': [r['type'] for r in eval_results]})

    # Step 3: PPO curriculum training loop with periodic CM checking
    pretrained_path = f'ppo_{args.task}_sb3/final_model.zip'  # default naming
    if not os.path.exists(pretrained_path):
        logging.warning(f"[AutoCaLC-CPDRL] Pretrained PPO not found at {pretrained_path}. Please train a base policy first.")
        return
    idx = 0
    cm_initials = [r['cm_score'] for r in eval_results]  # Store initial CM for each intervention
    while idx < len(eval_results):
        res = eval_results[idx]
        intervention = res['intervention']
        cm_initial = cm_initials[idx]
        logging.info(f"\n[AutoCaLC-CPDRL] === Intervention '{res['type']}' just got applied (rank {idx+1}) ===")
        if args.use_wandb:
            wandb.log({'curriculum/current_intervention': res['type'], 'curriculum/intervention_rank': idx+1})
        def make_env():
            return make_env_with_intervention(args.task, intervention, seed=args.seed)
        env = DummyVecEnv([make_env])
        ppo_config = {
            'gamma': 0.995,
            'n_steps': 2048,  # default for single env, can be adjusted
            'ent_coef': 0.02,
            'learning_rate': 2.5e-4,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'gae_lambda': 0.97,
            'batch_size': 512,
            'n_epochs': 15
        }
        policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 256])
        transfer_model = None
        total_timesteps = 0
        cm_reduction_streak = 0
        max_checks = 300
        required_streak = 10
        reduction_threshold = 0.10  # 10% reduction
        while cm_reduction_streak < required_streak and total_timesteps < args.timesteps:
            train_steps = 10000
            callback = CurriculumLoggingCallback(
                intervention_type=res['type'],
                intervention_idx=idx+1,
                wandb_enabled=args.use_wandb
            )
            transfer_model = ppo_train_on_intervention(
                pretrained_path, env, train_steps, args.log_dir, args.seed, ppo_config, policy_kwargs, transfer_model=transfer_model, callback=callback
            )
            total_timesteps += train_steps
            t_models, r_models, s_models, a_models = train_cm_models(env.envs[0], episodes=5, device=device)
            t_score, r_score, s_score, a_score = evaluate_cm_score(env.envs[0], t_models, r_models, s_models, a_models, episodes=5, device=device)
            cm_current = t_score + r_score + s_score + a_score
            reduction = (cm_initial - cm_current) / (abs(cm_initial) + 1e-8)
            logging.info(f"[AutoCaLC-CPDRL] Timesteps: {total_timesteps}, Current CM: {cm_current:.4f}, Reduction: {reduction:.2%} (Streak: {cm_reduction_streak})")
            if args.use_wandb:
                wandb.log({'curriculum/current_cm_score': cm_current, 'curriculum/cm_relative_drop': reduction, 'curriculum/timesteps': total_timesteps, 'curriculum/cm_reduction_streak': cm_reduction_streak})
            if reduction >= reduction_threshold:
                cm_reduction_streak += 1
                if cm_reduction_streak >= required_streak:
                    logging.info(f"[AutoCaLC-CPDRL] CM reduction > 10% for {required_streak} consecutive checks, moving to next intervention.")
                    if args.use_wandb:
                        wandb.log({'curriculum/cm_sufficient_reduction': True})
            else:
                cm_reduction_streak = 0
                logging.info(f"[AutoCaLC-CPDRL] CM reduction insufficient (< 10%), streak reset.")
                if args.use_wandb:
                    wandb.log({'curriculum/cm_sufficient_reduction': False})
            if total_timesteps >= args.timesteps:
                logging.info(f"[AutoCaLC-CPDRL] Reached max timesteps ({args.timesteps}), moving to next intervention.")
                if args.use_wandb:
                    wandb.log({'curriculum/max_timesteps_reached': True})
                break
        idx += 1
    logging.info("\n[AutoCaLC-CPDRL] Curriculum complete.")
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main() 