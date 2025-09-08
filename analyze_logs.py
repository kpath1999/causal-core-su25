#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from src.visualization import generate_comprehensive_visualizations

# Import the four required visualization functions
from visualize_baselines_v1 import (
    load_baseline_data, 
    process_for_plots, 
    plot_baseline_comparison,
    load_intervention_test_data, 
    plot_intervention_effectiveness,
    load_benchmark_results,
    plot_radar_chart
)

"""
COMMAND EXAMPLE:
python analyze_logs.py --log_dir logs --four_plots --baseline rnd --output_dir analysis/rnd_analysis
python analyze_logs.py --log_dir logs --four_plots --baseline greedy --output_dir analysis/greedy_analysis
"""

def main():
    parser = argparse.ArgumentParser(description="Generate the four required visualizations from curriculum learning logs")
    parser.add_argument('--log_dir', type=str, default='logs', help='Base directory containing log folders')
    parser.add_argument('--heuristics', type=str, help='Comma-separated list of heuristics to analyze (e.g., "greedy,cm,random")')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--four_plots', action='store_true', help='Generate the four required plots only')
    parser.add_argument('--comprehensive', action='store_true', help='Generate comprehensive visualizations')
    parser.add_argument('--baseline', type=str, help='Analyze only a specific baseline (greedy, cm, random, none, rnd, count, lpm, info)')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Output directory
    output_dir = args.output_dir or os.path.join(args.log_dir, 'comprehensive_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the four required plots if requested (default behavior)
    if args.four_plots or not args.comprehensive:
        try:
            logging.info("Generating four required visualization plots...")
            
            # Load baseline training data
            baseline_data = load_baseline_data(args.log_dir, args.baseline)
            if baseline_data:
                reward_df, success_df = process_for_plots(baseline_data)
                if not reward_df.empty and not success_df.empty:
                    plot_baseline_comparison(reward_df, success_df, output_dir)
            
            # Load intervention test data
            intervention_data = load_intervention_test_data(args.log_dir, args.baseline)
            if intervention_data:
                plot_intervention_effectiveness(intervention_data, output_dir)
            
            # Load benchmark results
            benchmark_data = load_benchmark_results(args.log_dir, args.baseline)
            if benchmark_data:
                plot_radar_chart(benchmark_data, output_dir)
                
            logging.info("Four plots generated successfully.")
        except Exception as e:
            logging.error(f"Error generating four plots: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Generate comprehensive visualizations if requested
    if args.comprehensive:
        # Parse heuristics list if provided
        heuristic_list = None
        if args.heuristics:
            heuristic_list = [h.strip() for h in args.heuristics.split(',')]
        
        try:
            generate_comprehensive_visualizations(args.log_dir, heuristic_list, output_dir)
            print(f"Comprehensive visualization generation complete. Results saved to {output_dir}")
        except Exception as e:
            logging.error(f"Error generating comprehensive visualizations: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()