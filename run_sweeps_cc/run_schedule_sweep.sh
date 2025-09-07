#!/bin/bash
set -euo pipefail

# Determine base directory (parent of run_sweeps_cc)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

# Required environment variables (must be set by the job script)
if [[ -z "${ALGORITHM:-}" || -z "${DATASET:-}" ]]; then
  echo "ERROR: Please export ALGORITHM and DATASET before running." >&2
  exit 1
fi

# DATA_SRC must point to the datasets root that contains subfolders like VLCS/, PACS/, ...
if [[ -z "${DATA_SRC:-}" ]]; then
  echo "ERROR: DATA_SRC is not set. Please export DATA_SRC to the dataset root (containing e.g. VLCS/, PACS/)." >&2
  echo "Example: export DATA_SRC=/project/$USER/dg_datasets" >&2
  exit 1
fi

# Validate that the expected dataset folder exists under DATA_SRC
if [[ ! -d "$DATA_SRC/$DATASET" ]]; then
  echo "ERROR: Dataset directory not found: $DATA_SRC/$DATASET" >&2
  echo "Ensure DATA_SRC points to the parent directory that contains $DATASET/." >&2
  exit 1
fi

echo "Using data_dir: $DATA_SRC"
echo "Dataset:       $DATASET"
echo "Algorithm:     $ALGORITHM"

# Schedule ranges (cosine). Defaults: ps 1->8, crop 1.0->0.5, gamma 1.0->0.2
PS_START=1
PS_END=8
CROP_START=1.0
CROP_END=0.7
GAMMA_START=0.6
GAMMA_END=0.3
SCHED_START_PCT=0.2
SCHED_END_PCT=1.0

# Base hparams
SHUFFLE_PATCH_SIZE=${PS_START}   # initial; scheduler will update
SHUFFLE_PATCH_EVERY=200
SHUFFLE_FN="shuffle_center_patches"
GRAD_INHIB_STRENGTH=${GAMMA_START}
DOT_PRODUCT_THRESH=0.4

# Other hparams
BATCH_SIZE=40
WEIGHT_DECAY=0.0003 #0.0005
LR=1e-5 #0.003
RESNET_DROPOUT=0.25

OUTPUT_DIR="${BASE_DIR}/results/${DATASET}/schedule_sweep/${ALGORITHM}/ps=${PS_START}to${PS_END}__crop=${CROP_START}to${CROP_END}__gamma=${GAMMA_START}to${GAMMA_END}__sched=${SCHED_START_PCT}to${SCHED_END_PCT}__every=${SHUFFLE_PATCH_EVERY}__fn=${SHUFFLE_FN}__dotth=${DOT_PRODUCT_THRESH}__bs=${BATCH_SIZE}__wd=${WEIGHT_DECAY}__lr=${LR}__dropout=${RESNET_DROPOUT}"
mkdir -p "${OUTPUT_DIR}"

python -m domainbed.scripts.sweep_schedule \
  launch \
  --datasets "${DATASET}" \
  --algorithms "${ALGORITHM}" \
  --n_hparams_from 0 \
  --n_hparams 1 \
  --n_trials 1 \
  --output_dir "${OUTPUT_DIR}" \
  --data_dir "${DATA_SRC}" \
  --command_launcher multi_gpu \
  --hparams "{\"shuffle_patch_size\": ${SHUFFLE_PATCH_SIZE}, \"shuffle_patch_every\": ${SHUFFLE_PATCH_EVERY}, \"shuffle_fn\": \"${SHUFFLE_FN}\", \"grad_inhib_strength\": ${GRAD_INHIB_STRENGTH}, \"dot_prod_thresh\": ${DOT_PRODUCT_THRESH}, \"batch_size\": ${BATCH_SIZE}, \"weight_decay\": ${WEIGHT_DECAY}, \"lr\": ${LR}, \"resnet_dropout\": ${RESNET_DROPOUT}, \"shuffle_patch_size_start\": ${PS_START}, \"shuffle_patch_size_end\": ${PS_END}, \"center_crop_ratio_start\": ${CROP_START}, \"center_crop_ratio_end\": ${CROP_END}, \"grad_inhib_strength_start\": ${GAMMA_START}, \"grad_inhib_strength_end\": ${GAMMA_END}, \"sched_start_pct\": ${SCHED_START_PCT}, \"sched_end_pct\": ${SCHED_END_PCT}}" \
  --single_test_envs \
  --skip_confirmation

python -m domainbed.scripts.collect_results --input_dir "${OUTPUT_DIR}"
