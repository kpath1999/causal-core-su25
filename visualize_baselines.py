#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import json

"""
COMMANDS TO RUN:

# Analyze all baselines (default)
python visualize_baselines.py --log_dir logs --output_dir paper_visualizations

# Analyze only the greedy baseline
python visualize_baselines.py --log_dir logs --output_dir greedy_analysis --baseline greedy

# Analyze only the cm baseline
python visualize_baselines.py --log_dir logs --output_dir cm_analysis --baseline cm

# Analyze only the random intervention baseline
python visualize_baselines.py --log_dir logs --output_dir random_analysis --baseline "random intervention"

# Analyze only the no intervention baseline
python visualize_baselines.py --log_dir logs --output_dir none_analysis --baseline "no intervention"

# Analyze only the RND baseline
python visualize_baselines.py --log_dir logs --output_dir rnd_analysis --baseline RND

# Generate AutoCaLC-specific plots along with all other visualizations
python visualize_baselines.py --log_dir logs --output_dir autocalc_analysis --autocalc_analysis

# Focus only on AutoCaLC analysis
python visualize_baselines.py --log_dir logs --output_dir autocalc_analysis --baseline autocalc

NEW VISUALIZATIONS GENERATED:
- focused_radar_plot.png/pdf: Radar chart focused on key curriculum strategies (autocalc, cm, random, greedy, none, pretrained)
- benchmark_summary_table.png/pdf: Tabular summary of benchmark performance across evaluation protocols  
- benchmark_summary_table.csv: CSV export of the benchmark summary for easy analysis
"""

def load_autocalc_pretrained_data(log_base_dir='logs'):
    """load validation data from autocalc runs with different pretrained teachers"""
    pretrained_data = {}

    # pattern to look for directories with pretrained teacher runs
    pretrained_patterns = [
        ('autocalc_teacher100k', '100k'),
        ('autocalc_teacher175k', '175k'),
        ('autocalc_teacher250k', '250k')
    ]

    for dir_pattern, label in pretrained_patterns:
        # find matching directories
        matching_dirs = glob.glob(os.path.join(log_base_dir, f"{dir_pattern}*"))

        if not matching_dirs:
            print(f"Warning: No logs found for {label} pretrained teacher")
            continue

        log_dir = matching_dirs[0]
        
        # load validation log
        validation_log_path = os.path.join(log_dir, 'validation_log.csv')
        if os.path.exists(validation_log_path):
            df = pd.read_csv(validation_log_path)

            # check for empty or all-null data
            if df.empty or df.isnull().all().all():
                print(f"Warning: Validation log for {label} pretrained teacher is empty or all nulls")
                continue

            # add pretrained steps identifier
            df['pretrain_steps'] = label
            pretrained_data[label] = df
        else:
            print(f"Warning: No validation_log.csv found for {label} pretrained teacher")
    
    return pretrained_data

def load_teacher_pretraining_data(log_base_dir='logs'):
    """Load teacher pretraining data to analyze learning progress"""
    pretraining_data = {}

    # pattern to look for teacher pretraining directories
    pretraining_patterns = [
        'teacher_pretrain_100k',
        'teacher_pretrain_175k',
        'teacher_pretrain_250k'
    ]

    for dir_pattern in pretraining_patterns:
        # find matching directories
        matching_dirs = glob.glob(os.path.join(log_base_dir, f"{dir_pattern}*"))

        if not matching_dirs:
            print(f"Warning: No logs found for {dir_pattern}")
            continue
            
        log_dir = matching_dirs[0]

        # load teacher pretraining log
        pretraining_log_path = os.path.join(log_dir, 'teacher_pretraining.csv')
        if os.path.exists(pretraining_log_path):
            df = pd.read_csv(pretraining_log_path)

            # extract steps from directory name
            steps = dir_pattern.split('_')[-1]
            pretraining_data[steps] = df
        else:
            # looking for alternate validation log
            pretrain_validation_dir = os.path.join(log_dir, "pretrain_validation")
            validation_log_path = os.path.join(pretrain_validation_dir, 'validation_log.csv')
            if os.path.exists(validation_log_path):
                df = pd.read_csv(validation_log_path)
                steps = dir_pattern.split('_')[-1]
                pretraining_data[steps] = df
            else:
                print(f"Warning: No validation data found for {dir_pattern}")
    
    return pretraining_data

def plot_pretrained_teacher_comparison(pretrained_data, output_dir='visualizations'):
    """Create plots comparing student performance with differently pretrained teachers"""
    if not pretrained_data:
        print("No pretrained teacher data found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create figure for validation reward comparison
    plt.figure(figsize=(12, 8))
    
    # Define colors for consistency - use distinct colors for the three teacher types
    color_map = {
        '100k': '#1f77b4',  # blue
        '175k': '#ff7f0e',  # orange
        '250k': '#2ca02c'   # green
    }
    
    # Plot each pretrained teacher's student performance
    for pretrain_steps, df in pretrained_data.items():
        # Ensure we have the required columns
        if 'stage' not in df.columns or 'validation_avg_reward' not in df.columns:
            print(f"Warning: Required columns missing for {pretrain_steps}")
            continue
        
        data = df.sort_values('stage')
        plt.plot(data['stage'], data['validation_avg_reward'], 
                 label=f'Teacher pretrained {pretrain_steps} steps', 
                 color=color_map.get(pretrain_steps, '#888888'), 
                 linewidth=2.5, marker='o', markersize=4)
        
        # Add confidence interval if std is available
        if 'validation_reward_std' in df.columns:
            plt.fill_between(data['stage'], 
                            data['validation_avg_reward'] - data['validation_reward_std'], 
                            data['validation_avg_reward'] + data['validation_reward_std'], 
                            alpha=0.2, color=color_map.get(pretrain_steps, '#888888'))
    
    plt.xlabel('Meta-Episode', fontsize=14)
    plt.ylabel('Validation Average Reward', fontsize=14)
    plt.title('Student Performance vs Meta-Episode\nwith Different Teacher Pretraining Steps', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'pretrained_teacher_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pretrained_teacher_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Pretrained teacher comparison plot saved to {output_dir}")

def plot_teacher_learning_curve(pretraining_data, validation_data, output_dir='visualizations'):
    """
    Plot max validation reward achieved by students under different teacher pretraining paradigms
    as a bar chart, with horizontal lines showing max rewards from other baselines
    """
    if not pretraining_data:
        print("No teacher pretraining data found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    plt.figure(figsize=(12, 8))
    
    # Extract max validation rewards for each pretrained teacher paradigm
    pretrained_max_rewards = {}
    teacher_labels = []
    max_rewards = []
    
    # Process pretrained teacher data
    for pretrain_steps, df in pretraining_data.items():
        if 'validation_avg_reward' in df.columns:
            max_reward = df['validation_avg_reward'].max()
            pretrained_max_rewards[pretrain_steps] = max_reward
            teacher_labels.append(f'AutoCaLC {pretrain_steps}')
            max_rewards.append(max_reward)
            print(f"Max reward for {pretrain_steps} pretrained teacher: {max_reward:.3f}")
    
    if not max_rewards:
        print("No valid pretrained teacher data found for bar plot")
        return
    
    # Create bar plot for pretrained teachers
    x_pos = np.arange(len(teacher_labels))
    bars = plt.bar(x_pos, max_rewards, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
                   alpha=0.8, width=0.6, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, reward) in enumerate(zip(bars, max_rewards)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{reward:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Calculate and plot horizontal lines for baseline max rewards
    baseline_max_rewards = {}
    colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_idx = 0
    
    for baseline_name, df in validation_data.items():
        if baseline_name.startswith('autocalc'):  # Skip autocalc baselines to avoid duplication
            continue
        
        if 'validation_avg_reward' in df.columns:
            max_reward = df['validation_avg_reward'].max()
            baseline_max_rewards[baseline_name] = max_reward
            
            # Plot horizontal line
            plt.axhline(y=max_reward, color=colors[color_idx % len(colors)], 
                       linestyle='--', linewidth=2, alpha=0.8,
                       label=f'{baseline_name}: {max_reward:.3f}')
            color_idx += 1
            print(f"Max reward for {baseline_name} baseline: {max_reward:.3f}")
    
    # Customize the plot
    plt.xlabel('Teacher Training Paradigm', fontsize=14)
    plt.ylabel('Maximum Validation Reward', fontsize=14)
    plt.title('Maximum Student Validation Reward\nAcross Different Teacher Pretraining Steps', fontsize=16)
    plt.xticks(x_pos, teacher_labels, fontsize=12)
    
    # Add legend for baseline horizontal lines
    if baseline_max_rewards:
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'teacher_max_reward_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'teacher_max_reward_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Teacher max reward comparison plot saved to {output_dir}")
    
    # Print summary
    print("\nSummary of Max Validation Rewards:")
    print("AutoCaLC Teacher Pretraining:")
    for steps, reward in pretrained_max_rewards.items():
        print(f"  {steps}: {reward:.3f}")
    print("Other Baselines:")
    for baseline, reward in baseline_max_rewards.items():
        print(f"  {baseline}: {reward:.3f}")

def load_baseline_data(log_base_dir='logs', target_baseline=None):
    """Load and process data from all baseline logs"""
    baseline_data = {}
    
    # Define mapping of directory patterns to baseline names
    baseline_dirs = {
        'greedy_replacement_sequencing_logs': 'greedy',
        'cm_replacement_sequencing_logs': 'cm',
        'random_replacement_sequencing_logs': 'random intervention',
        'none_sequencing_logs': 'no intervention',
        'rnd_sequencing_logs': 'RND',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info',
        'autocalc_qtable': 'autocalc'  # Added AutoCaLC
    }

    # Filter to specific baseline if requested
    if target_baseline:
        filtered_dirs = {k: v for k, v in baseline_dirs.items() if v == target_baseline}
        if not filtered_dirs:
            print(f"Error: Baseline '{target_baseline}' not found. Available baselines: {list(baseline_dirs.values())}")
            return {}
        baseline_dirs = filtered_dirs
    
    for dir_pattern, baseline_name in baseline_dirs.items():
        # Find all matching directories
        matching_dirs = glob.glob(os.path.join(log_base_dir, f"{dir_pattern}*"))
        
        if not matching_dirs:
            print(f"Warning: No logs found for {baseline_name} baseline")
            continue
            
        log_dir = matching_dirs[0]  # Use the first matching directory
        
        # Load training log
        training_log_path = os.path.join(log_dir, 'training_log.csv')
        if os.path.exists(training_log_path):
            df = pd.read_csv(training_log_path)
            
            # Add baseline identifier if not present
            if 'baseline_type' not in df.columns:
                df['baseline_type'] = baseline_name
                
            baseline_data[baseline_name] = df
        else:
            # Try to find SB3 logs
            sb3_logs = glob.glob(os.path.join(log_dir, "sb3_csv_logs_*", "progress.csv"))
            if sb3_logs:
                # Combine multiple stage logs if needed
                dfs = []
                for log_path in sb3_logs:
                    df = pd.read_csv(log_path)
                    # Extract stage from path
                    stage_str = os.path.basename(os.path.dirname(log_path)).split('_')[-1]
                    try:
                        df['stage'] = int(stage_str.replace('stage', ''))
                    except ValueError:
                        df['stage'] = 1
                    dfs.append(df)
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    combined_df['baseline_type'] = baseline_name
                    baseline_data[baseline_name] = combined_df
    
    return baseline_data

def load_validation_data(log_base_dir='logs', target_baseline=None):
    """Load and process validation data from all baseline logs"""
    validation_data = {}
    
    # Define mapping of directory patterns to baseline names
    baseline_dirs = {
        'greedy_replacement_sequencing_logs': 'greedy',
        'cm_replacement_sequencing_logs': 'cm',
        'random_replacement_sequencing_logs': 'random intervention',
        'none_sequencing_logs': 'no intervention',
        'rnd_sequencing_logs': 'RND',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info',
        'autocalc_qtable': 'autocalc'  # Added AutoCaLC
    }

    # Filter to specific baseline if requested
    if target_baseline:
        filtered_dirs = {k: v for k, v in baseline_dirs.items() if v == target_baseline}
        if not filtered_dirs:
            print(f"Error: Baseline '{target_baseline}' not found. Available baselines: {list(baseline_dirs.values())}")
            return {}
        baseline_dirs = filtered_dirs
    
    for dir_pattern, baseline_name in baseline_dirs.items():
        # Find all matching directories
        matching_dirs = glob.glob(os.path.join(log_base_dir, f"{dir_pattern}*"))
        
        if not matching_dirs:
            print(f"Warning: No logs found for {baseline_name} baseline")
            continue
            
        log_dir = matching_dirs[0]  # Use the first matching directory
        
        # Load validation log
        validation_log_path = os.path.join(log_dir, 'validation_log.csv')
        if os.path.exists(validation_log_path):
            df = pd.read_csv(validation_log_path)

            # check for empty or all-null data
            if df.empty or df.isnull().all().all():
                print(f"Warning: Validation log for {baseline_name} is empty or all nulls.")
                continue
            
            # Add baseline identifier if not present
            if 'baseline_type' not in df.columns:
                df['baseline_type'] = baseline_name
                
            validation_data[baseline_name] = df
        else:
            print(f"Warning: No validation_log.csv found for {baseline_name}")
    
    return validation_data

def process_validation_data_for_plots(validation_data):
    """Process the validation data for plotting rewards, success rates, and episode lengths by stage"""
    reward_data = []
    success_data = []
    length_data = []
    
    # Debug information
    print("\nValidation data processing diagnostics:")
    for baseline_name, df in validation_data.items():
        print(f"\n  Baseline: {baseline_name}")
        print(f"  - Dataframe shape: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        # Ensure we have the required columns
        if 'stage' not in df.columns:
            print(f"  - WARNING: No stage column found for {baseline_name}")
            continue
        
        # Process validation_avg_reward
        if 'validation_avg_reward' in df.columns and 'validation_reward_std' in df.columns:
            # Check for non-numeric or all NaN data
            reward_valid = df['validation_avg_reward'].notna().any() and df['validation_reward_std'].notna().any()
            print(f"  - Reward columns present: Yes, contains valid data: {reward_valid}")
            
            if reward_valid:
                reward_df = df[['stage', 'validation_avg_reward', 'validation_reward_std']].copy()
                reward_df.columns = ['stage', 'mean', 'std']
                reward_df['baseline'] = baseline_name
                reward_data.append(reward_df)
            else:
                print(f"  - WARNING: Reward columns for {baseline_name} contain only NaN values")
        else:
            missing = []
            if 'validation_avg_reward' not in df.columns:
                missing.append('validation_avg_reward')
            if 'validation_reward_std' not in df.columns:
                missing.append('validation_reward_std')
            print(f"  - Reward columns missing: {', '.join(missing)}")
        
        # Process validation_success_rate (similar checks)
        if 'validation_success_rate' in df.columns and 'validation_success_rate_std' in df.columns:
            success_valid = df['validation_success_rate'].notna().any() and df['validation_success_rate_std'].notna().any()
            if success_valid:
                success_df = df[['stage', 'validation_success_rate', 'validation_success_rate_std']].copy()
                success_df.columns = ['stage', 'mean', 'std']
                success_df['baseline'] = baseline_name
                success_data.append(success_df)
            else:
                print(f"  - WARNING: Success rate columns for {baseline_name} contain only NaN values")
                
        # Process validation_avg_length (similar checks)  
        if 'validation_avg_length' in df.columns and 'validation_length_std' in df.columns:
            length_valid = df['validation_avg_length'].notna().any() and df['validation_length_std'].notna().any()
            if length_valid:
                length_df = df[['stage', 'validation_avg_length', 'validation_length_std']].copy()
                length_df.columns = ['stage', 'mean', 'std']
                length_df['baseline'] = baseline_name
                length_data.append(length_df)
            else:
                print(f"  - WARNING: Length columns for {baseline_name} contain only NaN values")
    
    # Concatenate all data
    reward_df = pd.concat(reward_data, ignore_index=True) if reward_data else pd.DataFrame()
    success_df = pd.concat(success_data, ignore_index=True) if success_data else pd.DataFrame()
    length_df = pd.concat(length_data, ignore_index=True) if length_data else pd.DataFrame()
    
    print(f"\nSummary: Found reward data for {len(reward_data)}/{len(validation_data)} baselines")
    if reward_data:
        print(f"Baselines with valid reward data: {[df['baseline'].iloc[0] for df in reward_data]}")
    
    return reward_df, success_df, length_df

def load_intervention_test_data(log_base_dir='logs', target_baseline=None):
    """load and process intervention test data from all baseline logs"""
    intervention_data = {}

    # Define mapping of directory patterns to baseline names
    baseline_dirs = {
        'greedy_replacement_sequencing_logs': 'greedy',
        'cm_replacement_sequencing_logs': 'cm',
        'random_replacement_sequencing_logs': 'random intervention',
        'none_sequencing_logs': 'no intervention',
        'rnd_sequencing_logs': 'RND',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info',
        'autocalc_qtable': 'autocalc'  # Added AutoCaLC
    }

    # Filter to specific baseline if requested
    if target_baseline:
        filtered_dirs = {k: v for k, v in baseline_dirs.items() if v == target_baseline}
        if not filtered_dirs:
            print(f"Error: Baseline '{target_baseline}' not found. Available baselines: {list(baseline_dirs.values())}")
            return {}
        baseline_dirs = filtered_dirs

    for dir_pattern, baseline_name in baseline_dirs.items():
        # Find all matching directories
        matching_dirs = glob.glob(os.path.join(log_base_dir, f"{dir_pattern}*"))
        
        if not matching_dirs:
            print(f"Warning: No logs found for {baseline_name} baseline")
            continue
            
        log_dir = matching_dirs[0]  # Use the first matching directory
        
        # Load intervention test log
        intervention_log_path = os.path.join(log_dir, 'intervention_log.csv')
        if os.path.exists(intervention_log_path):
            df = pd.read_csv(intervention_log_path)
            
            # Add baseline identifier if not present
            if 'baseline_type' not in df.columns:
                df['baseline_type'] = baseline_name
                
            intervention_data[baseline_name] = df
    
    return intervention_data

def load_benchmark_results(log_base_dir='logs', target_baseline=None):
    """Load benchmark results from all baseline logs"""
    benchmark_data = {}
    
    # Define mapping of directory patterns to baseline names
    baseline_dirs = {
        'greedy_replacement_sequencing_logs': 'greedy',
        'cm_replacement_sequencing_logs': 'cm',
        'random_replacement_sequencing_logs': 'random intervention',
        'none_sequencing_logs': 'no intervention',
        'rnd_sequencing_logs': 'RND',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info',
        'pretrained_baseline_logs': 'pretrained',
        'autocalc_qtable': 'autocalc'  # Added AutoCaLC
    }

    # Filter to specific baseline if requested
    if target_baseline:
        filtered_dirs = {k: v for k, v in baseline_dirs.items() if v == target_baseline}
        if not filtered_dirs:
            print(f"Error: Baseline '{target_baseline}' not found. Available baselines: {list(baseline_dirs.values())}")
            return {}
        baseline_dirs = filtered_dirs
    
    for dir_pattern, baseline_name in baseline_dirs.items():
        # Find all matching directories
        matching_dirs = glob.glob(os.path.join(log_base_dir, f"{dir_pattern}*"))
        
        if not matching_dirs:
            print(f"Warning: No logs found for {baseline_name} baseline")
            continue
            
        log_dir = matching_dirs[0]  # Use the first matching directory
        
        # Load benchmark results
        benchmark_path = os.path.join(log_dir, 'benchmark_results.json')
        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r') as f:
                benchmark_results = json.load(f)
            benchmark_data[baseline_name] = benchmark_results
        else:
            print(f"Warning: No benchmark_results.json found for {baseline_name}")
    
    return benchmark_data

def plot_radar_chart(benchmark_data, output_dir='visualizations'):
    """Create radar plot for mean full integrated fractional success across baselines"""
    if not benchmark_data:
        print("No benchmark data found for radar chart")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract mean_full_integrated_fractional_success for each baseline
    baselines = []
    success_values = []
    
    for baseline_name, data in benchmark_data.items():
        # Check for both 'final_evals' (baselines) and 'benchmark_scores' (AutoCaLC)
        evals_key = None
        if 'final_evals' in data:
            evals_key = 'final_evals'
        elif 'benchmark_scores' in data:
            evals_key = 'benchmark_scores'
        
        if evals_key:
            # Get all protocol success rates
            protocol_values = []
            for protocol, metrics in data[evals_key].items():
                if 'mean_full_integrated_fractional_success' in metrics:
                    protocol_values.append(metrics['mean_full_integrated_fractional_success'])
            
            if protocol_values:
                baselines.append(baseline_name)
                success_values.append(protocol_values)
    
    if not baselines:
        print("No valid benchmark data found for radar chart")
        return
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Number of protocols
    num_protocols = len(success_values[0]) if success_values else 0
    protocol_names = [f'P{i}' for i in range(num_protocols)]
    
    # Angles for each protocol
    angles = np.linspace(0, 2 * np.pi, num_protocols, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the plot
    
    # Define colors for baselines
    colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))
    
    # Plot each baseline
    for i, (baseline, values) in enumerate(zip(baselines, success_values)):
        values_closed = values + [values[0]]  # Close the plot
        ax.plot(angles, values_closed, 'o-', linewidth=2, label=baseline, color=colors[i])
        ax.fill(angles, values_closed, alpha=0.25, color=colors[i])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(protocol_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean Full Integrated Fractional Success', labelpad=30)
    ax.set_title('Benchmark Performance Comparison Across Baselines', size=16, pad=30)
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_radar_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'benchmark_radar_plot.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Radar plot saved to {output_dir}")

def plot_focused_radar_chart(benchmark_data, output_dir='visualizations'):
    """Create focused radar plot with only key curriculum strategies"""
    if not benchmark_data:
        print("No benchmark data found for focused radar chart")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the key baselines we want to focus on
    key_baselines = ['autocalc', 'cm', 'random intervention', 'greedy', 'no intervention', 'pretrained']
    
    # Extract mean_full_integrated_fractional_success for key baselines only
    baselines = []
    success_values = []
    
    for baseline_name, data in benchmark_data.items():
        # Only include key baselines
        if baseline_name not in key_baselines:
            continue
            
        # Check for both 'final_evals' (baselines) and 'benchmark_scores' (AutoCaLC)
        evals_key = None
        if 'final_evals' in data:
            evals_key = 'final_evals'
        elif 'benchmark_scores' in data:
            evals_key = 'benchmark_scores'
        
        if evals_key:
            # Get all protocol success rates
            protocol_values = []
            for protocol, metrics in data[evals_key].items():
                if 'mean_full_integrated_fractional_success' in metrics:
                    protocol_values.append(metrics['mean_full_integrated_fractional_success'])
            
            if protocol_values:
                baselines.append(baseline_name)
                success_values.append(protocol_values)
    
    if not baselines:
        print("No valid benchmark data found for focused radar chart")
        return
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Number of protocols
    num_protocols = len(success_values[0]) if success_values else 0
    protocol_names = [f'Instance {i}' for i in range(num_protocols)]
    
    # Angles for each protocol
    angles = np.linspace(0, 2 * np.pi, num_protocols, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the plot
    
    # Define specific colors for key baselines for better readability
    color_map = {
        'autocalc': '#1f77b4',      # blue
        'cm': '#ff7f0e',            # orange  
        'random intervention': '#2ca02c',  # green
        'greedy': '#d62728',        # red
        'no intervention': '#9467bd',   # purple
        'pretrained': '#8c564b'     # brown
    }
    
    # Plot each baseline with specific colors
    for i, (baseline, values) in enumerate(zip(baselines, success_values)):
        values_closed = values + [values[0]]  # Close the plot
        color = color_map.get(baseline, plt.cm.tab10(i))
        ax.plot(angles, values_closed, 'o-', linewidth=3, label=baseline, color=color, markersize=6)
        ax.fill(angles, values_closed, alpha=0.15, color=color)
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(protocol_names, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean Full Integrated Fractional Success', labelpad=30, fontsize=12)
    ax.set_title('Key Curriculum Strategies: Benchmark Performance Comparison', size=16, pad=30, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend with better positioning
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'focused_radar_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'focused_radar_plot.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Focused radar plot saved to {output_dir}")

def create_benchmark_summary_table(benchmark_data, output_dir='visualizations'):
    """Create tabular summary of benchmark performance for all curriculum strategies"""
    if not benchmark_data:
        print("No benchmark data found for summary table")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Name mapping for display purposes
    display_name_map = {
        'random intervention': 'random',
        'no intervention': 'none'
    }
    
    # Collect data for table
    table_data = []
    
    for baseline_name, data in benchmark_data.items():
            
        # Check for both 'final_evals' (baselines) and 'benchmark_scores' (AutoCaLC)
        evals_key = None
        if 'final_evals' in data:
            evals_key = 'final_evals'
        elif 'benchmark_scores' in data:
            evals_key = 'benchmark_scores'
        
        if evals_key:
            # Use display name mapping for better readability
            display_name = display_name_map.get(baseline_name, baseline_name)
            row_data = {'Strategy': display_name}
            protocol_values = []
            
            # Extract protocol-specific scores
            for i, (protocol, metrics) in enumerate(data[evals_key].items()):
                if 'mean_full_integrated_fractional_success' in metrics:
                    score = metrics['mean_full_integrated_fractional_success']
                    row_data[f'Protocol {i}'] = f"{score:.3f}"
                    protocol_values.append(score)
            
            # Calculate average across all protocols
            if protocol_values:
                avg_score = np.mean(protocol_values)
                row_data['Average'] = f"{avg_score:.3f}"
                table_data.append(row_data)
    
    if not table_data:
        print("No valid benchmark data found for summary table")
        return
    
    # Create DataFrame and sort by average performance
    df = pd.DataFrame(table_data)
    df['avg_numeric'] = df['Average'].astype(float)
    df = df.sort_values('avg_numeric', ascending=False).drop('avg_numeric', axis=1)
    
    # Create a nice formatted table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # Highlight the best performer in each protocol
    for col_idx in range(1, len(df.columns)):
        col_name = df.columns[col_idx]
        if col_name != 'Strategy':
            max_val = df[col_name].astype(float).max()
            max_idx = df[col_name].astype(float).idxmax()
            row_idx = df.index.get_loc(max_idx) + 1
            table[(row_idx, col_idx)].set_facecolor('#FFD700')  # Gold color for best
    
    plt.title('Pushing Benchmark Performance Summary: All Curriculum Strategies', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'benchmark_summary_table.pdf'), bbox_inches='tight')
    plt.close()
    
    # Also save as CSV for easy access
    csv_path = os.path.join(output_dir, 'benchmark_summary_table.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Benchmark summary table saved to {output_dir}")
    print(f"CSV version saved as: {csv_path}")
    
    # Print summary to console
    print("\nBenchmark Performance Summary:")
    print("="*50)
    print(df.to_string(index=False))
    
    return df

def plot_intervention_effectiveness(intervention_data, output_dir='visualizations'):
    """Create intervention effectiveness bar chart (Reward only as main metric)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data for plotting
    all_dfs = []
    for baseline_name, df in intervention_data.items():
        # Skip empty dataframes
        if df.empty:
            print(f"  Skipping {baseline_name}: empty intervention data")
            continue
        df_copy = df.copy()
        df_copy['baseline'] = baseline_name
        all_dfs.append(df_copy)
    
    if not all_dfs:
        print("no intervention test data found - skipping intervention effectiveness plot")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Check if combined data is empty or missing required columns
    if combined_df.empty or 'stage' not in combined_df.columns:
        print("Insufficient intervention data - skipping intervention effectiveness plot")
        return
    
    # Get the latest meta-episode data for each baseline and intervention combination
    latest_stage_data = combined_df.groupby(['baseline', 'intervention_type']).apply(
        lambda x: x.loc[x['stage'].idxmax()]
    ).reset_index(drop=True)
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create the main intervention effectiveness plot (Average Reward)
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar chart for test_avg_reward
    pivot_data = latest_stage_data.pivot(index='intervention_type', columns='baseline', values='test_avg_reward')
    
    # Plot with better styling
    ax = pivot_data.plot(kind='bar', figsize=(14, 8), width=0.8)
    
    plt.xlabel('Intervention Type')
    plt.ylabel('Average Reward')
    plt.title('Intervention Test Effectiveness: Average Reward Across Baselines')
    plt.legend(title='Baseline', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figures
    plt.savefig(os.path.join(output_dir, 'intervention_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'intervention_effectiveness.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Intervention effectiveness plot saved to {output_dir}")

def compute_moving_average(data, window=5):
    """Compute moving average for a pandas Series"""
    return data.rolling(window=window, min_periods=1).mean()

def process_training_data_for_plots(baseline_data):
    """Process the training data for plotting rewards and success rates by stage using groupby"""
    reward_data = []
    success_data = []
    
    for baseline_name, df in baseline_data.items():
        # Ensure we have the required columns
        if 'stage' not in df.columns:
            print(f"Warning: No stage column found for {baseline_name}")
            continue
            
        # Use reward column and success column
        if 'reward' in df.columns and 'success' in df.columns:
            # Group by stage and calculate mean and std for rewards
            stage_rewards = df.groupby('stage')['reward'].agg(['mean', 'std']).reset_index()
            stage_rewards['baseline'] = baseline_name
            reward_data.append(stage_rewards)
            
            # Group by stage and calculate mean and std for success rates
            # Convert success to numeric if it's boolean
            if df['success'].dtype == bool:
                df['success'] = df['success'].astype(int)
            stage_success = df.groupby('stage')['success'].agg(['mean', 'std']).reset_index()
            stage_success['baseline'] = baseline_name
            success_data.append(stage_success)
        else:
            print(f"Warning: Required columns (reward, success) not found for {baseline_name}")
    
    if not reward_data:
        print("No training reward data found across baselines")
        return pd.DataFrame(), pd.DataFrame()
        
    if not success_data:
        print("No training success data found across baselines")
        return pd.DataFrame(), pd.DataFrame()
    
    reward_df = pd.concat(reward_data, ignore_index=True)
    success_df = pd.concat(success_data, ignore_index=True)
    
    return reward_df, success_df

def plot_training_comparison(reward_df, success_df, output_dir='visualizations'):
    """Create plots comparing baseline training performance with and without std dev"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data
    if reward_df.empty and success_df.empty:
        print("No training data available for baseline comparison")
        return
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Define colors for consistency
    if not reward_df.empty:
        unique_baselines = reward_df['baseline'].unique()
    elif not success_df.empty:
        unique_baselines = success_df['baseline'].unique()
    else:
        print("No baseline data found")
        return
        
    palette = sns.color_palette("tab10", n_colors=len(unique_baselines))
    color_map = dict(zip(sorted(unique_baselines), palette))
    
    # PLOT 1: Training Average reward with moving average (with std)
    if not reward_df.empty:
        plt.figure(figsize=(12, 8))
        
        for baseline in sorted(reward_df['baseline'].unique()):
            data = reward_df[reward_df['baseline'] == baseline].sort_values('stage')
            
            # Compute moving average
            data['moving_avg'] = compute_moving_average(data['mean'], window=5)
            
            plt.plot(data['stage'], data['moving_avg'], label=f'{baseline} (5-ep MA)', 
                     color=color_map[baseline], linewidth=2)
            
            # Add shaded region for standard deviation around moving average
            plt.fill_between(data['stage'], 
                             data['moving_avg'] - data['std'], 
                             data['moving_avg'] + data['std'], 
                             alpha=0.2, color=color_map[baseline])
        
        plt.xlabel('Meta-Episode')
        plt.ylabel('Average Reward')
        plt.title('Training: Average Reward Across Meta-Episodes by Baseline (5-episode Moving Average with std dev)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_reward_comparison_with_std.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'training_reward_comparison_with_std.pdf'))
        plt.close()
        
        # PLOT 1b: Training Average reward with moving average (without std)
        plt.figure(figsize=(12, 8))
        
        for baseline in sorted(reward_df['baseline'].unique()):
            data = reward_df[reward_df['baseline'] == baseline].sort_values('stage')
            
            # Compute moving average
            data['moving_avg'] = compute_moving_average(data['mean'], window=5)
            
            plt.plot(data['stage'], data['moving_avg'], label=f'{baseline} (5-ep MA)', 
                     color=color_map[baseline], linewidth=2)
        
        plt.xlabel('Meta-Episode')
        plt.ylabel('Average Reward')
        plt.title('Training: Average Reward Across Meta-Episodes by Baseline (5-episode Moving Average)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_reward_comparison.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'training_reward_comparison.pdf'))
        plt.close()
    
    # PLOT 2: Training Success rate frequency histogram
    if not success_df.empty:
        # Create frequency bins for success rate
        labels = ['0.0 (No overlap)', '(0.0, 0.99]', '1.0 (Perfect)']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        baseline_names = sorted(success_df['baseline'].unique())
        x_pos = np.arange(len(labels))
        width = 0.8 / len(baseline_names)
        
        for i, baseline in enumerate(baseline_names):
            baseline_data = success_df[success_df['baseline'] == baseline]['mean']
            
            # Calculate frequencies for each bin
            frequencies = []
            # Bin 1: Exactly 0.0
            count_zero = np.sum(baseline_data == 0.0)
            frequencies.append(count_zero)
            
            # Bin 2: (0.0, 0.99] - greater than 0 but less than 1
            count_middle = np.sum((baseline_data > 0.0) & (baseline_data <= 0.99))
            frequencies.append(count_middle)
            
            # Bin 3: Exactly 1.0 (perfect success)
            count_perfect = np.sum(baseline_data == 1.0)
            frequencies.append(count_perfect)
            
            # Plot bars
            ax.bar(x_pos + i * width, frequencies, width, 
                   label=baseline, color=color_map[baseline], alpha=0.8)
        
        ax.set_xlabel('Success Rate Bins')
        ax.set_ylabel('Frequency (Number of Meta-Episodes)')
        ax.set_title('Training: Success Rate Distribution by Baseline')
        ax.set_xticks(x_pos + width * (len(baseline_names) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_success_distribution.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'training_success_distribution.pdf'))
        plt.close()
    
    print(f"Training comparison plots saved to {output_dir}")

def plot_validation_comparison(reward_df, success_df, length_df, output_dir='visualizations'):
    """Create plots comparing baseline validation performance with and without std dev"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data
    if reward_df.empty and success_df.empty and length_df.empty:
        print("No validation data available for baseline comparison")
        return
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Define colors for consistency
    if not reward_df.empty:
        unique_baselines = reward_df['baseline'].unique()
    elif not success_df.empty:
        unique_baselines = success_df['baseline'].unique()
    elif not length_df.empty:
        unique_baselines = length_df['baseline'].unique()
    else:
        print("No baseline validation data found")
        return
        
    palette = sns.color_palette("tab10", n_colors=len(unique_baselines))
    color_map = dict(zip(sorted(unique_baselines), palette))

    # Add debugging for reward dataframe
    print("\nDebugging reward_df:")
    print(f"Shape: {reward_df.shape}")
    print(f"Unique baselines in reward_df: {sorted(reward_df['baseline'].unique())}")
    print(f"Expected baselines: {sorted(unique_baselines)}")
    print(f"Missing baselines: {set(unique_baselines) - set(reward_df['baseline'].unique())}")
    
    # PLOT 1: Validation Average reward with std (no moving average)
    if not reward_df.empty:
        # Export processed reward dataframe
        reward_df.to_csv('processed_validation_rewards.csv', index=False)
        print("Exported processed validation rewards to processed_validation_rewards.csv")

        plt.figure(figsize=(12, 8))
        print(f"Plotting validation rewards for baselines: {sorted(reward_df['baseline'].unique())}")
        
        # Loop through ALL unique baselines from any dataframe
        for baseline in sorted(unique_baselines):
            # Only plot if this baseline exists in reward_df
            if baseline in reward_df['baseline'].unique():
                data = reward_df[reward_df['baseline'] == baseline].sort_values('stage')
                plt.plot(data['stage'], data['mean'], label=f'{baseline}', 
                        color=color_map[baseline], linewidth=2)
                plt.fill_between(data['stage'], 
                                data['mean'] - data['std'], 
                                data['mean'] + data['std'], 
                                alpha=0.2, color=color_map[baseline])
            else:
                print(f"Warning: Baseline '{baseline}' has no reward data to plot")
        plt.xlabel('Meta-Episode')
        plt.ylabel('Average Reward')
        plt.title('Validation: Average Reward Across Meta-Episodes by Baseline (with std dev)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_reward_comparison_with_std.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'validation_reward_comparison_with_std.pdf'))
        plt.close()
        
        # PLOT 1b: Validation Average reward with moving average (without std)
        plt.figure(figsize=(12, 8))
        
        for baseline in sorted(reward_df['baseline'].unique()):
            data = reward_df[reward_df['baseline'] == baseline].sort_values('stage')
            
            # Compute moving average
            data['moving_avg'] = compute_moving_average(data['mean'], window=5)
            
            plt.plot(data['stage'], data['moving_avg'], label=f'{baseline} (5-ep MA)', 
                     color=color_map[baseline], linewidth=2)
        
        plt.xlabel('Meta-Episode')
        plt.ylabel('Average Reward')
        plt.title('Validation: Average Reward Across Meta-Episodes by Baseline (5-episode Moving Average)')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_reward_comparison.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'validation_reward_comparison.pdf'))
        plt.close()
    
    # PLOT 2: Validation Success rate frequency histogram
    if not success_df.empty:
        # Create frequency bins for success rate
        labels = ['0.0 (No overlap)', '(0.0, 0.99]', '1.0 (Perfect)']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        baseline_names = sorted(success_df['baseline'].unique())
        x_pos = np.arange(len(labels))
        width = 0.8 / len(baseline_names)
        
        for i, baseline in enumerate(baseline_names):
            baseline_data = success_df[success_df['baseline'] == baseline]['mean']
            
            # Calculate frequencies for each bin
            frequencies = []
            # Bin 1: Exactly 0.0
            count_zero = np.sum(baseline_data == 0.0)
            frequencies.append(count_zero)
            
            # Bin 2: (0.0, 0.99] - greater than 0 but less than 1
            count_middle = np.sum((baseline_data > 0.0) & (baseline_data <= 0.99))
            frequencies.append(count_middle)
            
            # Bin 3: Exactly 1.0 (perfect success)
            count_perfect = np.sum(baseline_data == 1.0)
            frequencies.append(count_perfect)
            
            # Plot bars
            ax.bar(x_pos + i * width, frequencies, width, 
                   label=baseline, color=color_map[baseline], alpha=0.8)
        
        ax.set_xlabel('Success Rate Bins')
        ax.set_ylabel('Frequency (Number of Meta-Episodes)')
        ax.set_title('Validation: Success Rate Distribution by Baseline')
        ax.set_xticks(x_pos + width * (len(baseline_names) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_success_distribution.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'validation_success_distribution.pdf'))
        plt.close()
    
    # PLOT 3: Validation Episode Length frequency histogram
    if not length_df.empty:
        # Create frequency bins for episode length
        labels = ['[0-100) (Efficient)', '[100-500) (Medium)', '500 (Max)']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        baseline_names = sorted(length_df['baseline'].unique())
        x_pos = np.arange(len(labels))
        width = 0.8 / len(baseline_names)
        
        for i, baseline in enumerate(baseline_names):
            baseline_data = length_df[length_df['baseline'] == baseline]['mean']
            
            # Calculate frequencies for each bin
            frequencies = []
            # Bin 1: [0-100) - efficient completion
            count_efficient = np.sum((baseline_data >= 0) & (baseline_data < 100))
            frequencies.append(count_efficient)
            
            # Bin 2: [100-500) - medium completion time
            count_medium = np.sum((baseline_data >= 100) & (baseline_data < 500))
            frequencies.append(count_medium)
            
            # Bin 3: Exactly 500 (max length)
            count_max = np.sum(baseline_data >= 500)
            frequencies.append(count_max)
            
            # Plot bars
            ax.bar(x_pos + i * width, frequencies, width, 
                   label=baseline, color=color_map[baseline], alpha=0.8)
        
        ax.set_xlabel('Episode Length Bins')
        ax.set_ylabel('Frequency (Number of Meta-Episodes)')
        ax.set_title('Validation: Episode Length Distribution by Baseline')
        ax.set_xticks(x_pos + width * (len(baseline_names) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_length_distribution.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'validation_length_distribution.pdf'))
        plt.close()
    
    print(f"Validation comparison plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualizations for training and validation data")
    parser.add_argument('--log_dir', type=str, default='logs', help='Base directory containing log folders')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--baseline', type=str, help='Analyze only a specific baseline (greedy, cm, "random intervention", "no intervention", RND, count, lpm, info, autocalc)')
    parser.add_argument('--autocalc_analysis', action='store_true', help='Generate AutoCaLC-specific analysis plots')
    
    args = parser.parse_args()
    
    if args.baseline:
        print(f"=== Generating Visualizations for {args.baseline.upper()} Baseline Only ===")
    else:
        print("=== Generating Comprehensive Visualizations for All Baselines ===")
    
    # 1. Load baseline training data and generate training plots
    print("1. Loading baseline training data...")
    baseline_data = load_baseline_data(args.log_dir, args.baseline)
    
    if baseline_data:
        print("   Processing training data for plots...")
        reward_df, success_df = process_training_data_for_plots(baseline_data)
        
        if not reward_df.empty or not success_df.empty:
            print("   Generating training plots...")
            plot_training_comparison(reward_df, success_df, args.output_dir)
        else:
            print("   Warning: No valid training data found for baseline comparison")
    
    # 2. Load validation data and generate validation plots
    print("2. Loading validation data...")
    validation_data = load_validation_data(args.log_dir, args.baseline)
    
    if validation_data:
        print("   Processing validation data for plots...")
        val_reward_df, val_success_df, val_length_df = process_validation_data_for_plots(validation_data)
    
        if not val_reward_df.empty or not val_success_df.empty or not val_length_df.empty:
            print("   Generating validation plots...")
            plot_validation_comparison(val_reward_df, val_success_df, val_length_df, args.output_dir)
        else:
            print("   Warning: No valid validation data found for baseline comparison")
    
    # 3. Load intervention test data and generate effectiveness plot
    print("3. Loading intervention test data...")
    intervention_data = load_intervention_test_data(args.log_dir, args.baseline)
    
    if intervention_data:
        print("   Generating intervention effectiveness plot...")
        plot_intervention_effectiveness(intervention_data, args.output_dir)
    else:
        print("   Warning: no intervention test data found")
    
    # 4. Load benchmark results and generate radar plot
    print("4. Loading benchmark results...")
    benchmark_data = load_benchmark_results(args.log_dir, args.baseline)
    
    if benchmark_data:
        print("   Generating radar plot...")
        plot_radar_chart(benchmark_data, args.output_dir)
        print("   Generating focused radar plot...")
        plot_focused_radar_chart(benchmark_data, args.output_dir)
        print("   Generating benchmark summary table...")
        create_benchmark_summary_table(benchmark_data, args.output_dir)
    else:
        print("   Warning: No benchmark results found")
    
    # additional AutoCaLC-specific analysis
    if args.autocalc_analysis or args.baseline == 'autocalc':
        print("5. Loading AutoCaLC pretrained teacher data...")
        pretrained_data = load_autocalc_pretrained_data(args.log_dir)

        if pretrained_data:
            print("   Generating pretrained teacher comparison plot...")
            plot_pretrained_teacher_comparison(pretrained_data, args.output_dir)
        else:
            print("   Warning: No AutoCaLC pretrained teacher data found")
        
        print("6. Loading teacher pretraining data...")
        pretraining_data = load_teacher_pretraining_data(args.log_dir)
        
        if pretraining_data:
            print("   Generating teacher learning curve plot...")
            plot_teacher_learning_curve(pretraining_data, validation_data, args.output_dir)
        else:
            print("   Warning: No teacher pretraining data found")
    
    print("=== Visualization generation complete! ===")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()