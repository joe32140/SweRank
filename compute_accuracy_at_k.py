#!/usr/bin/env python3
"""
Compute Accuracy@k metric for SweRank evaluation results.

Accuracy@k metric: Localization is successful if all relevant code locations 
are correctly identified within the top-k results.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import os


def extract_granularity_from_doc_id(doc_id):
    """
    Extract file, module (class), and function from a document ID.
    
    Args:
        doc_id: Document ID in format "file/path/module.py/ClassName/function_name"
                or "file/path/module.py/function_name"
    
    Returns:
        tuple: (file_path, module_class, function_name)
    """
    parts = doc_id.split('/')
    
    # Extract file path (everything before the last .py)
    file_parts = []
    for i, part in enumerate(parts):
        if part.endswith('.py'):
            file_parts.append(part)
            remaining_parts = parts[i+1:]
            break
        file_parts.append(part)
    
    file_path = '/'.join(file_parts)
    
    # Extract module/class and function
    if len(remaining_parts) == 1:
        # Format: file.py/function_name
        module_class = None
        function_name = remaining_parts[0]
    elif len(remaining_parts) == 2:
        # Format: file.py/ClassName/function_name
        module_class = remaining_parts[0]
        function_name = remaining_parts[1]
    else:
        # Handle nested classes or complex paths
        module_class = '/'.join(remaining_parts[:-1])
        function_name = remaining_parts[-1]
    
    return file_path, module_class, function_name


def load_ground_truth_from_dataset(dataset_dir, instance_id):
    """
    Load ground truth relevance judgments from the dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory
        instance_id: Instance ID to load ground truth for
    
    Returns:
        set: Set of relevant document IDs
    """
    # Find the dataset instance directory
    instance_dirs = []
    for item in os.listdir(dataset_dir):
        if instance_id in item:
            instance_dirs.append(item)
    
    if not instance_dirs:
        print(f"Warning: No dataset directory found for instance {instance_id}")
        return set()
    
    # Use the first matching directory (there should be only one)
    instance_dir = instance_dirs[0]
    qrels_file = os.path.join(dataset_dir, instance_dir, 'qrels', 'test.tsv')
    
    if not os.path.exists(qrels_file):
        print(f"Warning: No qrels file found for instance {instance_id}")
        return set()
    
    relevant_docs = set()
    with open(qrels_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], parts[2]
                if float(score) > 0:  # Relevant document
                    relevant_docs.add(doc_id)
    
    return relevant_docs


def compute_accuracy_at_k(results_file, dataset_dir, k_values=[1, 3, 5, 10, 20]):
    """
    Compute Accuracy@k for different granularities.
    
    Args:
        results_file: Path to the results JSON file
        dataset_dir: Path to the dataset directory containing ground truth
        k_values: List of k values to compute accuracy for
    
    Returns:
        dict: Accuracy results for different granularities and k values
    """
    instances = []
    
    # Handle JSONL format (one JSON object per line)
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    instance = json.loads(line)
                    instances.append(instance)
                except json.JSONDecodeError:
                    # Try to load the entire file as a single JSON
                    f.seek(0)
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            instances = data
                        else:
                            instances = [data]
                        break
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        return {}
    
    results = defaultdict(lambda: defaultdict(list))
    
    for instance in instances:
        instance_id = instance.get('instance_id')
        if not instance_id:
            continue
            
        # Load ground truth from dataset directory
        gt_docs = load_ground_truth_from_dataset(dataset_dir, instance_id)
        
        if not gt_docs:
            continue
        
        # Get retrieved documents
        retrieved_docs = instance.get('docs', [])
        
        # Organize ground truth by granularity
        gt_locations = {
            'file': set(),
            'module': set(),
            'function': set()
        }
        
        for doc_id in gt_docs:
            file_path, module_class, function_name = extract_granularity_from_doc_id(doc_id)
            
            gt_locations['file'].add(file_path)
            if module_class:
                gt_locations['module'].add(f"{file_path}/{module_class}")
            gt_locations['function'].add(doc_id)
        
        # Compute accuracy for each granularity and k value
        for granularity in ['file', 'module', 'function']:
            gt_set = gt_locations[granularity]
            
            if not gt_set:
                # Skip if no ground truth for this granularity
                continue
            
            for k in k_values:
                # Get top-k retrieved documents
                top_k_docs = retrieved_docs[:k]
                
                # Extract locations at current granularity from retrieved docs
                retrieved_locations = set()
                for doc_id in top_k_docs:
                    file_path, module_class, function_name = extract_granularity_from_doc_id(doc_id)
                    
                    if granularity == 'file':
                        retrieved_locations.add(file_path)
                    elif granularity == 'module' and module_class:
                        retrieved_locations.add(f"{file_path}/{module_class}")
                    elif granularity == 'function':
                        retrieved_locations.add(doc_id)
                
                # Check if all ground truth locations are in top-k
                success = gt_set.issubset(retrieved_locations)
                results[granularity][k].append(1 if success else 0)
    
    # Compute average accuracy
    final_results = {}
    for granularity in results:
        final_results[granularity] = {}
        for k in results[granularity]:
            accuracies = results[granularity][k]
            if accuracies:
                final_results[granularity][f'Acc@{k}'] = np.mean(accuracies)
                final_results[granularity][f'Count@{k}'] = len(accuracies)
            else:
                final_results[granularity][f'Acc@{k}'] = 0.0
                final_results[granularity][f'Count@{k}'] = 0
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Compute Accuracy@k metrics for SweRank results')
    parser.add_argument('results_file', help='Path to the results JSON file')
    parser.add_argument('--dataset_dir', default='datasets', 
                        help='Path to the dataset directory containing ground truth (default: datasets)')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 3, 5, 10, 20],
                        help='K values to compute accuracy for (default: 1 3 5 10 20)')
    parser.add_argument('--output', help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Compute accuracy
    results = compute_accuracy_at_k(args.results_file, args.dataset_dir, args.k_values)
    
    # Print results
    print(f"Accuracy@k Results for {Path(args.results_file).name}")
    print("=" * 60)
    
    for granularity in ['file', 'module', 'function']:
        if granularity in results:
            print(f"\n{granularity.upper()} Level:")
            print("-" * 20)
            for k in args.k_values:
                acc_key = f'Acc@{k}'
                count_key = f'Count@{k}'
                if acc_key in results[granularity]:
                    acc = results[granularity][acc_key]
                    count = results[granularity][count_key]
                    print(f"  Acc@{k:2d}: {acc:.3f} ({count} instances)")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
