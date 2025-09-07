import collections
import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

def remove_key(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def recursive_freeze(obj):
    if isinstance(obj, dict):
        return frozenset((key, recursive_freeze(val)) for key, val in obj.items())
    elif isinstance(obj, list):
        return tuple(recursive_freeze(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(recursive_freeze(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_freeze(item) for item in obj)
    else:
        return obj

def merge_records(records):
    merged_records = []
    args_set = set()  # Store unique args dictionaries

    # Group records by unique 'args' dictionaries
    for record in records:
        args = record['args'].copy()
        args.pop('holdout_fraction', None)  # Remove 'holdout_fraction' from comparison
        args_key = recursive_freeze(args)
        args_set.add(args_key)

    # Merge records with the same 'args' except for 'holdout_fraction'
    for args_key in args_set:
        args_dict = dict(args_key)
        filtered_records = [record for record in records if dict(recursive_freeze(remove_key(record['args'],'holdout_fraction'))) == args_dict]
        merged_record = {}
        for record in filtered_records:
            merged_record.update(record)
        merged_records.append(merged_record)
    return Q(merged_records)

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def filter_records_by_algorithm(records, algorithm=None):
    """Filter records based on algorithm"""
    if not algorithm:
        return records
        
    filtered_records = []
    
    for record in records:
        # Check if the record matches the algorithm filter
        if record['args']['algorithm'] == algorithm:
            filtered_records.append(record)
                
    return filtered_records

def get_all_hparam_keys(records):
    """Extract all unique hyperparameter keys from all records"""
    all_keys = set()
    for record in records:
        if 'hparams' in record:
            all_keys.update(record['hparams'].keys())
    return sorted(list(all_keys))

def group_records_by_all_hparams(records):
    """Group records by unique combinations of all hyperparameters"""
    groups = {}
    
    for record in records:
        # Create a key tuple with all hyperparameter values
        hparams = record.get('hparams', {})
        
        # Create a sorted tuple of (key, value) pairs for consistent hashing
        key_values = []
        for key in sorted(hparams.keys()):
            hashable_value = recursive_freeze(hparams[key])
            key_values.append((key, hashable_value))
        
        # Convert list to tuple for dictionary key
        key_tuple = tuple(key_values)
        
        # Add the record to the appropriate group
        if key_tuple not in groups:
            groups[key_tuple] = []
        groups[key_tuple].append(record)
    
    # Convert dictionary to list of groups
    grouped_records = []
    for key_tuple, group_records in groups.items():
        # Convert frozen values back to original values for display
        hparam_values_dict = {}
        for key, frozen_value in key_tuple:
            # Convert frozen structures back to original form for display
            if isinstance(frozen_value, frozenset):
                # Try to convert back to list or dict
                try:
                    if all(isinstance(item, tuple) and len(item) == 2 for item in frozen_value):
                        hparam_values_dict[key] = dict(frozen_value)
                    else:
                        hparam_values_dict[key] = list(frozen_value)
                except:
                    hparam_values_dict[key] = frozen_value
            elif isinstance(frozen_value, tuple):
                hparam_values_dict[key] = list(frozen_value)
            else:
                hparam_values_dict[key] = frozen_value
        
        grouped_records.append({
            'records': group_records,
            'hparam_values': hparam_values_dict
        })
    
    return grouped_records

def print_results_tables(records, selection_method, latex):
    """Given filtered records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records)

    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])

    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

def filter_incomplete_records(records, input_dir):
    """Filter out records from directories that don't have a 'done' file."""
    complete_records = []
    incomplete_count = 0
    
    for record in records:
        # Extract the directory path from the record
        if 'args' in record and 'output_dir' in record['args']:
            record_dir = record['args']['output_dir']
        else:
            # If we can't determine the directory, include the record
            complete_records.append(record)
            continue
            
        # Check if 'done' file exists
        done_file = os.path.join(record_dir, 'done')
        if os.path.exists(done_file):
            complete_records.append(record)
        else:
            incomplete_count += 1
            
    return complete_records, incomplete_count

def calculate_average_accuracy(records_subset, selection_method):
    """Calculate the average accuracy across all datasets for a group of records."""
    grouped_records = reporting.get_grouped_records(records_subset)
    
    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])

    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # Get dataset names
    dataset_names = Q(records_subset).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]
    
    # Get algorithm names
    alg_names = Q(records_subset).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])
    
    # Calculate average accuracy across all datasets and algorithms
    all_averages = []
    for algorithm in alg_names:
        for dataset in dataset_names:
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            if len(trial_averages) > 0:
                avg_acc = np.mean(list(trial_averages))
                all_averages.append(avg_acc)
    
    return np.mean(all_averages) if all_averages else 0.0

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization results collection grouped by ALL hyperparameters")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--auto_lr", action="store_true")
    parser.add_argument("--algorithm", type=str, help="Filter results by algorithm name")
    parser.add_argument("--ignore_incomplete", action="store_true", help="Ignore directories without a 'done' file")
    args = parser.parse_args()

    results_file = "results_all_hparams.tex" if args.latex else "results_all_hparams.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)
    print(f"Total records before filtering: {len(records)}")
    
    # Filter out incomplete directories if requested
    if args.ignore_incomplete:
        original_count = len(records)
        records, incomplete_count = filter_incomplete_records(records, args.input_dir)
        print(f"Filtered out {incomplete_count} incomplete records (missing 'done' file)")
    
    # Filter records by algorithm if specified
    filtered_records = filter_records_by_algorithm(records, args.algorithm)
    print(f"Records after filtering: {len(filtered_records)}")
    
    if len(filtered_records) == 0:
        print("No records match the specified filters.")
        sys.exit(0)
    
    # Get all unique hyperparameter keys
    all_hparam_keys = get_all_hparam_keys(filtered_records)
    print(f"All hyperparameter keys found: {all_hparam_keys}")
    
    # Group records by all hyperparameters
    grouped_records = group_records_by_all_hparams(filtered_records)
    print(f"Number of unique hyperparameter combinations: {len(grouped_records)}")
    
    # Calculate average accuracy for each group and sort by it
    iid_selection_method = model_selection.IIDAccuracySelectionMethod
    for group in grouped_records:
        group['avg_accuracy'] = calculate_average_accuracy(group['records'], iid_selection_method)
    
    # Sort groups by average accuracy in descending order
    grouped_records.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    print(f"Groups sorted by average accuracy (IIDAccuracySelectionMethod)")
    
    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{DomainBed results grouped by ALL hyperparameters}")
        if args.algorithm:
            print(f"\\textbf{{Algorithm:}} {args.algorithm}\\\\")
        if args.ignore_incomplete:
            print(f"\\textbf{{Note:}} Incomplete experiments were ignored.")
    else:
        print("DomainBed results grouped by ALL hyperparameters")
        if args.algorithm:
            print(f"Algorithm: {args.algorithm}")
        if args.ignore_incomplete:
            print(f"Note: Incomplete experiments were ignored.")

    if args.auto_lr:
        SELECTION_METHODS = [model_selection.IIDAutoLRAccuracySelectionMethod]
    else:
        SELECTION_METHODS = [
            model_selection.IIDAccuracySelectionMethod,
            model_selection.LeaveOneOutSelectionMethod,
            model_selection.OracleSelectionMethod,
        ]

    # For each group of records, print results tables
    for i, group in enumerate(grouped_records):
        records_subset = group['records']
        hparam_values = group['hparam_values']
        avg_accuracy = group['avg_accuracy']
        
        if args.latex:
            print()
            print("\\section{Hyperparameter Configuration %d (Avg Acc: %.2f)}" % (i+1, avg_accuracy))
            print("\\begin{itemize}")
            for key, value in hparam_values.items():
                print(f"\\item \\textbf{{{key}:}} {value}")
            print("\\end{itemize}")
        else:
            print("\n" + "="*80)
            print(f"Hyperparameter Configuration {i+1} (Average Accuracy: {avg_accuracy:.2f})")
            for key, value in hparam_values.items():
                print(f"  {key}: {value}")
            print("="*80)
        
        for selection_method in SELECTION_METHODS:
            if args.latex:
                print()
                print("\\subsection{{Model selection: {}}}".format(
                    selection_method.name))
            print_results_tables(records_subset, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
