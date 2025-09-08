d#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Amend failed sweep jobs - find incomplete jobs and relaunch them
"""

import argparse
import os
import shutil
import json
import ast
import re
from pathlib import Path
import tqdm

from domainbed.scripts.sweep import Job
from domainbed import command_launchers
from domainbed import datasets

# This is the list of boolean flags that use action='store_true'
BOOLEAN_FLAGS = ['skip_model_save', 'save_model_every_checkpoint']

# Create a custom Job class that correctly handles boolean flags
class AmendJob(Job):
    def __init__(self, train_args, sweep_output_dir, train_module="domainbed.scripts.train"):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        # Respect the original training module (train, train_schedule, train_autolr, ...)
        command = ['python', '-m', train_module]
        for k, v in sorted(self.train_args.items()):
            # Skip boolean flags that are False
            if k in BOOLEAN_FLAGS and not v:
                continue
                
            if k in BOOLEAN_FLAGS and v:
                # Add boolean flags without values
                command.append(f'--{k}')
            elif isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
                command.append(f'--{k} {v}')
            elif isinstance(v, str):
                v = shlex.quote(v)
                command.append(f'--{k} {v}')
            else:
                command.append(f'--{k} {v}')
                
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

# Import these after defining AmendJob to avoid circular imports
import hashlib
import copy
import shlex

def parse_test_envs(test_envs_str):
    """Parse test environments string to list of integers"""
    if '[' in test_envs_str and ']' in test_envs_str:
        # Extract content between brackets
        content = re.search(r'\[(.*?)\]', test_envs_str).group(1)
        if content:
            return [int(x.strip()) for x in content.split(',')]
        else:
            return []
    else:
        # Handle single environment case
        try:
            return [int(test_envs_str)]
        except ValueError:
            # If we can't parse it as a single int, return empty list
            return []

def parse_args_from_out_txt(out_txt_path):
    """Parse arguments from out.txt file"""
    args_dict = {}
    hparams_dict = {}
    
    section = None
    in_hparams = False
    
    with open(out_txt_path, 'r') as f:
        lines = f.readlines()
    
    # First find the Args and HParams sections
    args_start = -1
    hparams_start = -1
    for i, line in enumerate(lines):
        if line.strip() == "Args:":
            args_start = i + 1
        elif line.strip() == "HParams:":
            hparams_start = i + 1
            break
    
    # Extract Args section (stop when we hit HParams, or on blank/traceback/path)
    if args_start > 0:
        i = args_start
        while i < len(lines):
            line = lines[i].strip()
            # Stop if we reached HParams header
            if (hparams_start > 0) and (i >= hparams_start - 1):
                break
            # Stop at tracebacks or obvious path/error starts
            if line.startswith("/") or line.startswith("Traceback"):
                break
            # Skip empty lines
            if not line:
                i += 1
                continue
            if ":" in line:
                try:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "test_envs":
                        value = parse_test_envs(value)
                    elif key in ["checkpoint_freq", "steps"]:
                        # Skip None values for integer parameters
                        if value.lower() == "none":
                            i += 1
                            continue
                        else:
                            try:
                                value = int(value)
                            except ValueError:
                                # If it's not a valid integer, skip this parameter
                                i += 1
                                continue
                    elif key in ["save_model_every_checkpoint", "skip_model_save"]:
                        value = (value.lower() == "true")
                    elif key in ["holdout_fraction", "uda_holdout_fraction"]:
                        if value.lower() == "none":
                            i += 1
                            continue
                        try:
                            value = float(value)
                        except ValueError:
                            i += 1
                            continue
                    elif key in ["hparams_seed", "seed", "trial_seed"]:
                        if value.lower() == "none":
                            i += 1
                            continue
                        try:
                            value = int(value)
                        except ValueError:
                            i += 1
                            continue
                    
                    # Keep other keys as raw strings
                    args_dict[key] = value
                except Exception as e:
                    print(f"Error parsing args line: {line} - {e}")
            i += 1
    
    # Extract HParams section (stop when we hit a line that looks like a path/traceback)
    if hparams_start > 0:
        i = hparams_start
        while i < len(lines):
            line = lines[i].strip()
            
            # Stop if we hit what looks like a path or error message
            if line.startswith("/") or line.startswith("Traceback") or not line:
                break
                
            if ":" in line:
                try:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Skip if key contains a path or looks suspicious
                    if "/" in key or "." in key or len(key) > 30:
                        i += 1
                        continue
                        
                    # Skip if no value
                    if not value:
                        i += 1
                        continue
                        
                    # Convert values to appropriate types
                    if key in ["weight_decay", "lr", "resnet_dropout", "grad_inhib_strength"]:
                        try:
                            # Handle scientific notation
                            value = float(value)
                        except ValueError:
                            # If it's a string like "0.5", convert it
                            try:
                                value = float(value.strip('"\''))
                            except ValueError:
                                i += 1
                                continue
                    elif key in ["batch_size", "linear_steps", "shuffle_patch_every", "shuffle_patch_size"]:
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = int(value.strip('"\''))
                            except ValueError:
                                i += 1
                                continue
                    elif key in ["class_balanced", "data_augmentation", "nonlinear_classifier", "resnet18", 
                                "clip_inhib", "dinov2", "dinov2_inhib", "dinov2_inhib_base",
                                "freeze_bn", "lars", "resnet50_augmix", "vit", "vit_attn_tune"]:
                        # Handle boolean values that might be stored as strings
                        if isinstance(value, str):
                            if value.lower() == "true" or value == "True":
                                value = True
                            elif value.lower() == "false" or value == "False":
                                value = False
                            # Keep as string if it's already quoted
                            elif (value.startswith('"') and value.endswith('"')) or \
                                 (value.startswith("'") and value.endswith("'")):
                                value = value
                            else:
                                value = str(value)
                        
                    hparams_dict[key] = value
                except Exception as e:
                    print(f"Error parsing hparams line: {line} - {e}")
            i += 1
    
    # If we have hparams values, add them to args dict
    if hparams_dict:
        args_dict['hparams'] = json.dumps(hparams_dict)

    # Clean up unusable or misleading values
    if 'hparams' in args_dict and str(args_dict['hparams']).strip().lower() in {"none", "null", ""}:
        # Remove '--hparams None' to avoid JSON parse errors in train scripts
        args_dict.pop('hparams')

    return args_dict

def detect_train_module(out_txt_path):
    """Infer which training module created this out.txt.
    Returns a module path usable with `python -m`.
    Defaults to domainbed.scripts.train if unsure.
    """
    try:
        with open(out_txt_path, 'r') as f:
            text = f.read()
        if 'domainbed/scripts/train_schedule.py' in text:
            return 'domainbed.scripts.train_schedule'
        if 'domainbed/scripts/train_autolr.py' in text or 'train_autolr' in text:
            return 'domainbed.scripts.train_autolr'
    except Exception:
        pass
    return 'domainbed.scripts.train'

def infer_dataset_from_path(path_str):
    """Infer dataset name by scanning the path for any known dataset names."""
    candidates = [d for d in datasets.DATASETS if d in path_str]
    # Filter out debug datasets (to avoid false positives)
    candidates = [c for c in candidates if not c.startswith('Debug')]
    if len(candidates) == 1:
        return candidates[0]
    return None

def find_incomplete_jobs(root_dir):
    """Find all job directories without a 'done' file"""
    incomplete_dirs = []
    
    for job_dir in os.listdir(root_dir):
        job_path = os.path.join(root_dir, job_dir)
        
        # Skip if not a directory
        if not os.path.isdir(job_path):
            continue
            
        # Check if 'done' file exists
        done_file = os.path.join(job_path, 'done')
        if not os.path.exists(done_file):
            out_file = os.path.join(job_path, 'out.txt')
            
            # Check if out.txt exists
            if os.path.exists(out_file):
                incomplete_dirs.append(job_path)
    
    return incomplete_dirs

def prepare_job_dir(job_dir):
    """Clean up and prepare the job directory for relaunch"""
    # We'll preserve out.txt for reference but remove other files
    for item in os.listdir(job_dir):
        item_path = os.path.join(job_dir, item)
        if os.path.isfile(item_path) and item != 'out.txt':
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def main():
    parser = argparse.ArgumentParser(description='Amend failed sweep jobs')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing job directories')
    parser.add_argument('--command_launcher', type=str, required=True,
                        help='Command launcher to use (local, slurm, etc.)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data_dir if missing in out.txt')
    parser.add_argument('--train_module', type=str, default=None,
                        help='Force training module (e.g., domainbed.scripts.train_schedule). If omitted, auto-detect per job.')
    parser.add_argument('--delete_dir', action='store_true',
                        help='Delete and recreate job directories instead of emptying them')
    parser.add_argument('--dry_run', action='store_true',
                        help='Don\'t actually launch jobs, just show what would be done')
    parser.add_argument('--verbose', action='store_true',
                        help='Print more detailed information during processing')
    parser.add_argument('--show_parsed_hparams', action='store_true',
                        help='Print the parsed hparams for each job')
    args = parser.parse_args()
    
    root_dir = args.input_dir
    
    # Find incomplete jobs
    print(f"Scanning {root_dir} for incomplete jobs...")
    incomplete_dirs = find_incomplete_jobs(root_dir)
    print(f"Found {len(incomplete_dirs)} incomplete jobs.")
    
    if not incomplete_dirs:
        print("No incomplete jobs to amend.")
        return
    
    # Process each incomplete job
    jobs_to_launch = []
    
    for job_dir in tqdm.tqdm(incomplete_dirs, desc="Preparing jobs"):
        out_txt_path = os.path.join(job_dir, 'out.txt')
        
        # Parse arguments from out.txt
        try:
            job_args = parse_args_from_out_txt(out_txt_path)
            
            # Get parent directory of the job directory
            sweep_output_dir = os.path.dirname(job_dir)
            
            if args.verbose:
                print(f"\nJob directory: {job_dir}")
                
            if args.show_parsed_hparams and 'hparams' in job_args:
                print(f"\nParsed hparams for {os.path.basename(job_dir)}:")
                parsed_hparams = json.loads(job_args['hparams'])
                for k, v in sorted(parsed_hparams.items()):
                    print(f"  {k}: {v}")
            
            # Fill in missing critical args
            # dataset
            if 'dataset' not in job_args or not job_args.get('dataset') or job_args.get('dataset') == 'None':
                inferred_ds = infer_dataset_from_path(job_dir)
                if inferred_ds:
                    job_args['dataset'] = inferred_ds
                    if args.verbose:
                        print(f"Inferred dataset from path: {inferred_ds}")

            # data_dir
            if ('data_dir' not in job_args or not job_args.get('data_dir') or job_args.get('data_dir') == 'None') and args.data_dir:
                job_args['data_dir'] = args.data_dir
                if args.verbose:
                    print(f"Using provided data_dir: {args.data_dir}")

            # If deleting, remove and recreate the directory
            if args.delete_dir:
                if not args.dry_run:
                    shutil.rmtree(job_dir)
                    os.makedirs(job_dir, exist_ok=True)
            else:
                # Otherwise, clean up the directory
                if not args.dry_run:
                    prepare_job_dir(job_dir)
            
            # Decide which train module to use
            train_module = args.train_module or detect_train_module(out_txt_path)
            if args.verbose:
                print(f"Train module: {train_module}")

            # Create a job object with our custom class that handles boolean flags
            job = AmendJob(job_args, sweep_output_dir, train_module=train_module)
            
            if args.verbose:
                print(f"Command: {job.command_str}")
                
            jobs_to_launch.append(job)
            
        except Exception as e:
            print(f"Error processing {job_dir}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"Prepared {len(jobs_to_launch)} jobs for relaunch.")
    
    # Launch the jobs
    if not args.dry_run and jobs_to_launch:
        print(f"Launching {len(jobs_to_launch)} jobs...")
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(jobs_to_launch, launcher_fn)
        print("Done!")
    elif args.dry_run:
        print("Dry run - not launching jobs.")
        for job in jobs_to_launch[:5]:  # Show first 5 jobs as examples
            print(f"Would launch: {job.command_str}")
        if len(jobs_to_launch) > 5:
            print(f"... and {len(jobs_to_launch) - 5} more jobs")

if __name__ == "__main__":
    main()
