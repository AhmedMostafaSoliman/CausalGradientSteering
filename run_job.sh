#!/bin/bash
#SBATCH --account=rrg-mshehata      # Replace with your actual account
#SBATCH --time=24:00:00                 # Walltime (days-hours:minutes:seconds) - 4 hours
#SBATCH --nodes=1                    # Number of nodes (you want 4 GPUs on 1 node)
#SBATCH --ntasks-per-node=1          # Number of tasks per node (usually 1 for GPU jobs)
#SBATCH --cpus-per-task=8           # CPUs per task (adjust based on your needs)
#SBATCH --mem=128G                   # Memory per node (adjust based on your needs)
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:4       # or a MIG profile from above
#SBATCH --job-name=cgs              # Name of the job
#SBATCH --output=output_%j.out      # Output file (stdout) - %j is the job ID
#SBATCH --error=error_%j.err        # Error file (stderr) - %j is the job ID

set -euo pipefail

module load python/3.11                   # pick the version you need
module load cuda/12.6                 

echo "Host: $HOSTNAME"

export ALGORITHM="ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule"
export DATASET="PACS"
export SWEEP=1
# --- storage roots on Alliance ---
export PROJECT=/project/def-mshehata-ab
export SCRATCH=/home/ahmedm04/scratch
export CODE_SRC=$SCRATCH/projects/CausalGradientSteering
# Persistent datasets root (contains subfolders like VLCS/, PACS/, ...)
export DATA_SRC_PERSIST=$PROJECT/$USER/dg_datasets
export VENV_ACT=$PROJECT/$USER/venvs/cgsenv/bin/activate

# --- Slurm tmpdir (node-local SSD for speed) ---
export SLURM_TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
echo "SLURM_TMPDIR=$SLURM_TMPDIR"

# Stage code/data to node-local SSD
mkdir -p "$SLURM_TMPDIR/CausalGradientSteering"
rsync -a --delete "$CODE_SRC/" "$SLURM_TMPDIR/CausalGradientSteering/"
# Stage only the selected dataset to node-local SSD under a common root
DATA_SRC_LOCAL="$SLURM_TMPDIR/CausalGradientSteering/dg_datasets"
mkdir -p "$DATA_SRC_LOCAL/$DATASET"
rsync -a "$DATA_SRC_PERSIST/$DATASET/" "$DATA_SRC_LOCAL/$DATASET/"

# Activate your env
source "$VENV_ACT"

# Cache dirs on local SSD (fast, ephemeral)
export TORCH_HOME="$SLURM_TMPDIR/torch_cache"
export HF_HOME="$SLURM_TMPDIR/hf_cache"
mkdir -p "$TORCH_HOME/hub/checkpoints" "$HF_HOME"

# (optional) seed pre-downloaded checkpoints from /project (only if present)
SEED_TORCH_SRC="$PROJECT/$USER/pretrained_models/torch/hub/checkpoints"
SEED_HF_SRC="$PROJECT/$USER/pretrained_models/huggingface"
if [[ -d "$SEED_TORCH_SRC" ]]; then
  echo "Seeding torch hub checkpoints from: $SEED_TORCH_SRC"
  rsync -a "$SEED_TORCH_SRC/" "$TORCH_HOME/hub/checkpoints/"
else
  echo "No torch hub checkpoints found at: $SEED_TORCH_SRC (skipping)"
fi
if [[ -d "$SEED_HF_SRC" ]]; then
  echo "Seeding HF cache from: $SEED_HF_SRC"
  rsync -a "$SEED_HF_SRC/" "$HF_HOME/"
else
  echo "No HF cache found at: $SEED_HF_SRC (skipping)"
fi

cd "$SLURM_TMPDIR/CausalGradientSteering"
echo "Running from: $(pwd)"



# Ensure the sweep script sees the dataset root on local SSD
export DATA_SRC="$DATA_SRC_LOCAL"

srun sh run_sweeps/run_schedule_sweep_explore.sh

# Archive results only to project to avoid duplicates
DEST_PROJ="$PROJECT/$USER/CausalGradient/results/${SLURM_JOB_ID}"
echo "Archiving results to project at: $DEST_PROJ/"
mkdir -p "$DEST_PROJ"
rsync -a results/ "$DEST_PROJ/"
