#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from src.visualization import generate_comprehensive_visualizations

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualization reports from curriculum learning logs")
    parser.add_argument('--log_dir', type=str, default='logs', help='Base directory containing log folders')
    parser.add_argument('--heuristics', type=str, help='Comma-separated list of heuristics to analyze (e.g., "greedy,cm,random")')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse heuristics list if provided
    heuristic_list = None
    if args.heuristics:
        heuristic_list = [h.strip() for h in args.heuristics.split(',')]
    
    # Generate visualizations
    try:
        generate_comprehensive_visualizations(args.log_dir, heuristic_list)
        print(f"Visualization generation complete. Results saved to {os.path.join(args.log_dir, 'comprehensive_visualizations')}")
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()