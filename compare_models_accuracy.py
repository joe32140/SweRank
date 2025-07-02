#!/usr/bin/env python3
"""
Compare Accuracy@k results across different models.
"""

import json
import argparse
from pathlib import Path
from compute_accuracy_at_k import compute_accuracy_at_k


def compare_models(results_files, dataset_dir, k_values=[1, 3, 5, 10, 20]):
    """
    Compare accuracy results across multiple models.
    
    Args:
        results_files: List of (model_name, results_file_path) tuples
        dataset_dir: Path to dataset directory
        k_values: List of k values to compute accuracy for
    
    Returns:
        dict: Comparison results
    """
    all_results = {}
    
    for model_name, results_file in results_files:
        print(f"Computing accuracy for {model_name}...")
        results = compute_accuracy_at_k(results_file, dataset_dir, k_values)
        all_results[model_name] = results
    
    return all_results


def print_comparison_table(all_results, k_values):
    """Print a comparison table of results."""
    
    granularities = ['file', 'module', 'function']
    
    for granularity in granularities:
        print(f"\n{granularity.upper()} Level Accuracy@k")
        print("=" * 60)
        
        # Print header
        header = f"{'Model':<25}"
        for k in k_values:
            header += f"Acc@{k:<3}"
        header += "Count"
        print(header)
        print("-" * len(header))
        
        # Print results for each model
        for model_name, results in all_results.items():
            if granularity in results:
                row = f"{model_name:<25}"
                for k in k_values:
                    acc_key = f'Acc@{k}'
                    if acc_key in results[granularity]:
                        acc = results[granularity][acc_key]
                        row += f"{acc:.3f} "
                    else:
                        row += "N/A   "
                
                # Add count
                count_key = f'Count@{k_values[0]}'  # Use first k for count
                if count_key in results[granularity]:
                    count = results[granularity][count_key]
                    row += f"{count:>5}"
                else:
                    row += "  N/A"
                
                print(row)


def main():
    parser = argparse.ArgumentParser(description='Compare Accuracy@k across models')
    parser.add_argument('--dataset_dir', default='datasets', 
                        help='Path to dataset directory (default: datasets)')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 3, 5, 10, 20],
                        help='K values to compute accuracy for (default: 1 3 5 10 20)')
    parser.add_argument('--output', help='Output JSON file to save comparison results')
    
    args = parser.parse_args()
    
    # Define the models and their result files
    results_files = [
        ("SageLite-s", "results/model=SageLite-s_dataset=swe-bench-lite_split=test_level=function_evalmode=default_results.json"),
        ("CodeRankEmbed", "results/model=CodeRankEmbed_dataset=swe-bench-lite_split=test_level=function_evalmode=default_results.json"),
        ("Reason-ModernColBERT", "results/model=Reason-ModernColBERT_dataset=swe-bench-lite_split=test_level=function_evalmode=fixed_results.json"),
        # ("finetuned-Reason-ModernColBERT", "results/model=checkpoint-180_dataset=swe-bench-lite_split=test_level=function_evalmode=fixed_results.json"),
        ("finetuned-Reason-ModernColBERT", "results/model=checkpoint-90_dataset=swe-bench-lite_split=test_level=function_evalmode=fixed_results.json"),
    ]
    
    # Check if files exist
    existing_files = []
    for model_name, file_path in results_files:
        if Path(file_path).exists():
            existing_files.append((model_name, file_path))
        else:
            print(f"Warning: Results file not found for {model_name}: {file_path}")
    
    if not existing_files:
        print("No valid results files found!")
        return
    
    # Compare models
    all_results = compare_models(existing_files, args.dataset_dir, args.k_values)
    
    # Print comparison table
    print("\nModel Comparison - Accuracy@k Results")
    print("=" * 80)
    print_comparison_table(all_results, args.k_values)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()
