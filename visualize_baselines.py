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
"""


def load_baseline_data(log_base_dir='logs', target_baseline=None):
    """Load and process data from all baseline logs"""
    baseline_data = {}
    
    # Define mapping of directory patterns to baseline names
    baseline_dirs = {
        'greedy_replacement_sequencing_logs': 'greedy',
        'cm_replacement_sequencing_logs': 'cm',
        'random_replacement_sequencing_logs': 'random',
        'none_sequencing_logs': 'none',
        'rnd_sequencing_logs': 'rnd',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info'
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

def load_intervention_test_data(log_base_dir='logs', target_baseline=None):
    """load and process intervention test data from all baseline logs"""
    intervention_data = {}

    # Define mapping of directory patterns to baseline names
    baseline_dirs = {
        'greedy_replacement_sequencing_logs': 'greedy',
        'cm_replacement_sequencing_logs': 'cm',
        'random_replacement_sequencing_logs': 'random',
        'none_sequencing_logs': 'none',
        'rnd_sequencing_logs': 'rnd',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info'
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
        'random_replacement_sequencing_logs': 'random',
        'none_sequencing_logs': 'none',
        'rnd_sequencing_logs': 'rnd',
        'count_sequencing_logs': 'count',
        'lpm_sequencing_logs': 'lpm',
        'info_sequencing_logs': 'info'
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
        if 'final_evals' in data:
            # Get all protocol success rates
            protocol_values = []
            for protocol, metrics in data['final_evals'].items():
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
    colors = plt.cm.Set3(np.linspace(0, 1, len(baselines)))
    
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

def plot_intervention_effectiveness(intervention_data, output_dir='visualizations'):
    """Create intervention effectiveness bar chart (Reward only as main metric)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the data for plotting
    all_dfs = []
    for baseline_name, df in intervention_data.items():
        df_copy = df.copy()
        df_copy['baseline'] = baseline_name
        all_dfs.append(df_copy)
    
    if not all_dfs:
        print("No intervention test data found")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
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

def plot_baseline_comparison(reward_df, success_df, output_dir='visualizations'):
    """Create plots comparing baseline performance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data
    if reward_df.empty and success_df.empty:
        print("No data available for baseline comparison")
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
        
    palette = sns.color_palette("husl", n_colors=len(unique_baselines))
    color_map = dict(zip(sorted(unique_baselines), palette))
    
    # PLOT 1: Average reward across meta-episodes
    if not reward_df.empty:
        plt.figure(figsize=(12, 8))
        
        for baseline in sorted(reward_df['baseline'].unique()):
            data = reward_df[reward_df['baseline'] == baseline]
            plt.plot(data['stage'], data['mean'], label=baseline, 
                     color=color_map[baseline], linewidth=2)
            
            # Add shaded region for standard deviation
            plt.fill_between(data['stage'], 
                             data['mean'] - data['std'], 
                             data['mean'] + data['std'], 
                             alpha=0.2, color=color_map[baseline])
        
        plt.xlabel('Meta-Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Across Meta-Episodes by Baseline')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baseline_reward_comparison.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'baseline_reward_comparison.pdf'))
        plt.close()
    
    # PLOT 2: Success rate across meta-episodes
    if not success_df.empty:
        plt.figure(figsize=(12, 8))
        
        for baseline in sorted(success_df['baseline'].unique()):
            data = success_df[success_df['baseline'] == baseline]
            plt.plot(data['stage'], data['mean'], label=baseline, 
                     color=color_map[baseline], linewidth=2)
            
            # Add shaded region for standard deviation
            plt.fill_between(data['stage'], 
                             np.maximum(0, data['mean'] - data['std']),  # Ensure not below 0
                             np.minimum(1, data['mean'] + data['std']),  # Ensure not above 1
                             alpha=0.2, color=color_map[baseline])
        
        plt.xlabel('Meta-Episode')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Across Meta-Episodes by Baseline')
        plt.ylim(0, 1)  # Success rate should be between 0 and 1
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baseline_success_comparison.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'baseline_success_comparison.pdf'))
        plt.close()
    
    print(f"Baseline comparison plots saved to {output_dir}")


def process_for_plots(baseline_data):
    """Process the data for plotting rewards and success rates by stage using groupby"""
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
        print("No reward data found across baselines")
        return pd.DataFrame(), pd.DataFrame()
        
    if not success_data:
        print("No success data found across baselines")
        return pd.DataFrame(), pd.DataFrame()
    
    reward_df = pd.concat(reward_data, ignore_index=True)
    success_df = pd.concat(success_data, ignore_index=True)
    
    return reward_df, success_df

def plot_baseline_comparison(reward_df, success_df, output_dir='visualizations'):
    """Create plots comparing baseline performance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Define colors for consistency
    palette = sns.color_palette("husl", n_colors=len(reward_df['baseline'].unique()))
    color_map = dict(zip(sorted(reward_df['baseline'].unique()), palette))
    
    # PLOT 1: Average reward across meta-episodes
    plt.figure(figsize=(12, 8))
    
    for baseline in sorted(reward_df['baseline'].unique()):
        data = reward_df[reward_df['baseline'] == baseline]
        plt.plot(data['stage'], data['mean'], label=baseline, 
                 color=color_map[baseline], linewidth=2)
        
        # Add shaded region for standard deviation
        plt.fill_between(data['stage'], 
                         data['mean'] - data['std'], 
                         data['mean'] + data['std'], 
                         alpha=0.2, color=color_map[baseline])
    
    plt.xlabel('Meta-Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Across Meta-Episodes by Baseline')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_reward_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'baseline_reward_comparison.pdf'))
    plt.close()
    
    # PLOT 2: Success rate across meta-episodes
    plt.figure(figsize=(12, 8))
    
    for baseline in sorted(success_df['baseline'].unique()):
        data = success_df[success_df['baseline'] == baseline]
        plt.plot(data['stage'], data['mean'], label=baseline, 
                 color=color_map[baseline], linewidth=2)
        
        # Add shaded region for standard deviation
        plt.fill_between(data['stage'], 
                         np.maximum(0, data['mean'] - data['std']),  # Ensure not below 0
                         np.minimum(1, data['mean'] + data['std']),  # Ensure not above 1
                         alpha=0.2, color=color_map[baseline])
    
    plt.xlabel('Meta-Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Across Meta-Episodes by Baseline')
    plt.ylim(0, 1)  # Success rate should be between 0 and 1
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_success_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'baseline_success_comparison.pdf'))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate the four required visualizations")
    parser.add_argument('--log_dir', type=str, default='logs', help='Base directory containing log folders')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--baseline', type=str, help='Analyze only a specific baseline (greedy, cm, random, none, rnd, count, lpm, info)')
    
    args = parser.parse_args()
    
    if args.baseline:
        print(f"=== Generating Visualizations for {args.baseline.upper()} Baseline Only ===")
    else:
        print("=== Generating Four Required Visualizations for All Baselines ===")
    
    # 1. Load baseline training data and generate reward/success plots
    print("1. Loading baseline training data...")
    baseline_data = load_baseline_data(args.log_dir)
    
    if baseline_data:
        print("   Processing data for training plots...")
        reward_df, success_df = process_for_plots(baseline_data)
        
        if not reward_df.empty and not success_df.empty:
            print("   Generating training reward and success rate plots...")
            plot_baseline_comparison(reward_df, success_df, args.output_dir)
        else:
            print("   Warning: No valid training data found for baseline comparison")
    
    # 2. Load intervention test data and generate effectiveness plot
    print("2. Loading intervention test data...")
    intervention_data = load_intervention_test_data(args.log_dir)
    
    if intervention_data:
        print("   Generating intervention effectiveness plot...")
        plot_intervention_effectiveness(intervention_data, args.output_dir)
    else:
        print("   Warning: No intervention test data found")
    
    # 3. Load benchmark results and generate radar plot
    print("3. Loading benchmark results...")
    benchmark_data = load_benchmark_results(args.log_dir)
    
    if benchmark_data:
        print("   Generating radar plot...")
        plot_radar_chart(benchmark_data, args.output_dir)
    else:
        print("   Warning: No benchmark results found")
    
    print("=== Visualization generation complete! ===")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()