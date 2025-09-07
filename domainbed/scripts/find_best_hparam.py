# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Find the best hyperparameter dictionary averaged across all test domains.

Example usage:
python -u -m domainbed.scripts.find_best_hparam \
    --input_dir /home/ahmedm04/projects/CausalGradient/results/PACS/ablations/ERM_GRAD_INHIB_FAST_conditional_normalized2_dp_thresh_layer/sweep1 \
    --algorithm ERM_GRAD_INHIB_FAST_conditional_normalized2_dp_thresh_layer \
    --dataset PACS \
    --selection_method IIDAccuracySelectionMethod
"""

import argparse
import json
import sys
from collections import defaultdict, Counter

import numpy as np

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import my_model_selection as model_selection  # Use custom model selection module
from domainbed.lib.query import Q


def get_best_hparams_per_test_env(records, dataset, algorithm, selection_method_class):
    """Get best hyperparameters for each test environment."""
    grouped_records = reporting.get_grouped_records(records)
    
    # Filter records for the specific dataset and algorithm
    filtered_records = grouped_records.filter(
        lambda r: r['dataset'] == dataset and r['algorithm'] == algorithm
    )
    
    best_hparams_per_env = {}
    
    # Get number of test environments for this dataset
    num_envs = datasets.num_environments(dataset)
    
    for test_env in range(num_envs):
        env_records = filtered_records.filter(lambda r: r['test_env'] == test_env)
        
        if not env_records:
            continue
            
        # Get best hyperparameters for this test environment
        env_best_hparams = []
        
        for group in env_records:
            trial_seed = group['trial_seed']
            best_hparams = selection_method_class.hparams_accs(group['records'])
            
            if best_hparams:
                # Get the best hyperparameter set (first one with highest accuracy)
                best_acc_dict, best_hparam_records = best_hparams[0]
                hparams = best_hparam_records[0]['hparams']
                # Extract the validation accuracy used for selection
                val_acc = best_acc_dict['val_acc']
                test_acc = best_acc_dict['test_acc']
                env_best_hparams.append((val_acc, test_acc, hparams, trial_seed))
        
        if env_best_hparams:
            best_hparams_per_env[test_env] = env_best_hparams
    
    return best_hparams_per_env


def aggregate_hyperparameters(best_hparams_per_env):
    """Aggregate hyperparameters across test environments."""
    # Collect all hyperparameter keys
    all_keys = set()
    for test_env, hparam_list in best_hparams_per_env.items():
        for val_acc, test_acc, hparams, trial_seed in hparam_list:
            all_keys.update(hparams.keys())
    
    aggregated_hparams = {}
    
    for key in all_keys:
        values_with_weights = []
        
        # Collect values and their weights (test accuracies) across all test environments
        for test_env, hparam_list in best_hparams_per_env.items():
            for val_acc, test_acc, hparams, trial_seed in hparam_list:
                if key in hparams:
                    # Use test accuracy as the weight for aggregation
                    values_with_weights.append((hparams[key], test_acc))
        
        if not values_with_weights:
            continue
        
        # Check if all values are numerical
        numerical_values = []
        non_numerical_values = []
        
        for value, weight in values_with_weights:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numerical_values.append((value, weight))
            else:
                non_numerical_values.append((value, weight))
        
        # If all values are numerical, compute weighted average
        if numerical_values and not non_numerical_values:
            total_weight = sum(weight for _, weight in numerical_values)
            if total_weight > 0:
                weighted_avg = sum(value * weight for value, weight in numerical_values) / total_weight
                aggregated_hparams[key] = weighted_avg
            else:
                # Fallback to simple average
                aggregated_hparams[key] = sum(value for value, _ in numerical_values) / len(numerical_values)
        
        # If values are non-numerical or mixed, use weighted voting
        else:
            all_values = numerical_values + non_numerical_values
            
            # Count weighted votes for each unique value
            value_weights = defaultdict(float)
            for value, weight in all_values:
                value_weights[value] += weight
            
            # Find the value(s) with maximum weight
            max_weight = max(value_weights.values())
            best_values = [value for value, weight in value_weights.items() if weight == max_weight]
            
            if len(best_values) == 1:
                aggregated_hparams[key] = best_values[0]
            else:
                # Store all tied values
                aggregated_hparams[key] = best_values
    
    return aggregated_hparams


def resolve_ties(aggregated_hparams):
    """Generate all possible hyperparameter combinations when there are ties."""
    # Find keys with multiple values (ties)
    tied_keys = []
    tied_values = []
    
    for key, value in aggregated_hparams.items():
        if isinstance(value, list) and len(value) > 1:
            tied_keys.append(key)
            tied_values.append(value)
    
    if not tied_keys:
        return [aggregated_hparams]
    
    # Generate all combinations
    from itertools import product
    
    combinations = list(product(*tied_values))
    result_hparams = []
    
    for combo in combinations:
        hparams_copy = aggregated_hparams.copy()
        for i, key in enumerate(tied_keys):
            hparams_copy[key] = combo[i]
        result_hparams.append(hparams_copy)
    
    return result_hparams


def print_results(best_hparams_candidates, dataset, algorithm, selection_method_name):
    """Print the results in a readable format."""
    print(f"\nBest hyperparameters for {algorithm} on {dataset}")
    print(f"Selection method: {selection_method_name}")
    print(f"Number of candidate hyperparameter sets: {len(best_hparams_candidates)}")
    print("=" * 80)
    
    for i, hparams in enumerate(best_hparams_candidates, 1):
        print(f"\nCandidate {i}:")
        print("-" * 40)
        for key, value in sorted(hparams.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Print as JSON for easy copying
        print("\nAs JSON:")
        print(json.dumps(hparams, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find best hyperparameters averaged across test domains")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing sweep results")
    parser.add_argument('--dataset', required=True,
                        help="Dataset name")
    parser.add_argument('--algorithm', required=True,
                        help="Algorithm name")
    parser.add_argument('--selection_method', default='IIDAccuracySelectionMethod',
                        choices=['IIDAccuracySelectionMethod', 'LeaveOneOutSelectionMethod', 'OracleSelectionMethod'],
                        help="Model selection method")
    parser.add_argument('--output_file', default=None,
                        help="Optional output file to save results")
    args = parser.parse_args()

    # Load records
    records = reporting.load_records(args.input_dir)
    print(f"Total records loaded: {len(records)}")

    # Get selection method class
    selection_method_class = getattr(model_selection, args.selection_method)
    
    # Get best hyperparameters for each test environment
    best_hparams_per_env = get_best_hparams_per_test_env(
        records, args.dataset, args.algorithm, selection_method_class
    )
    
    if not best_hparams_per_env:
        print(f"No records found for {args.algorithm} on {args.dataset}")
        sys.exit(1)
    
    print(f"Found best hyperparameters for {len(best_hparams_per_env)} test environments")
    
    # Show per-environment results
    for test_env, hparam_list in best_hparams_per_env.items():
        print(f"\nTest environment {test_env}: {len(hparam_list)} trials")
        for val_acc, test_acc, hparams, trial_seed in hparam_list:
            print(f"  Trial {trial_seed}: val_acc = {val_acc:.4f}, test_acc = {test_acc:.4f}")
    
    # Aggregate hyperparameters
    aggregated_hparams = aggregate_hyperparameters(best_hparams_per_env)
    
    # Resolve ties to get all possible combinations
    best_hparams_candidates = resolve_ties(aggregated_hparams)
    
    # Print results
    print_results(best_hparams_candidates, args.dataset, args.algorithm, args.selection_method)
    
    # Optionally save to file
    if args.output_file:
        output_data = {
            'dataset': args.dataset,
            'algorithm': args.algorithm,
            'selection_method': args.selection_method,
            'best_hparams_candidates': best_hparams_candidates,
            'per_env_details': {
                str(test_env): [
                    {'val_accuracy': val_acc, 'test_accuracy': test_acc, 'trial_seed': trial_seed, 'hparams': hparams}
                    for val_acc, test_acc, hparams, trial_seed in hparam_list
                ]
                for test_env, hparam_list in best_hparams_per_env.items()
            }
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        
        print(f"\nResults saved to: {args.output_file}")
