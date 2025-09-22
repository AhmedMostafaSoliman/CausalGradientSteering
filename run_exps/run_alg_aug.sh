#!/usr/bin/env bash
set -euo pipefail

########################################
# >>> EDIT THESE TOP-LEVEL KNOBS <<<
########################################
DATASET="PACS"
DATA_DIR="/home/ahmedm04/dg_datasets"
OUTPUT_DIR="./results"

ALGO="ERM_GRAD_INHIB_FAST_conditional_normalized2_dp_thresh_layer"  # base algorithm
AUG2="IPMix"                            # the augmentation to pair with Original
AUG_FLAGS=(--augmentations Original "${AUG2}")  # don't edit!

# Number of trials
N_TRIALS=3
N_HPARAMS=1

# Optional: extra hparams for the aug (IPMix example). [No need to edit.]
#EXTRA_HP='{"ipmix_mixing_set_path": "/abs/path/to/ifs", "ipmix_k": 3, "ipmix_t": 3, "ipmix_severity": 2}'
EXTRA_HP="{}"
########################################
# Compose a readable tag for outputs
########################################
AUG_STACK="Original_${AUG2}"
TAG="${DATASET}_${ALGO}_${AUG_STACK}"

########################################
# Kick off the sweep (single-source DG)
#  - --single_train_envs makes each env the sole training env, tests on others
#  - --command_launcher multi_gpu balances jobs across available GPUs
########################################
python -m domainbed.scripts.sweep launch \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}/${TAG}" \
  --algorithms "${ALGO}" \
  --datasets "${DATASET}" \
  --n_hparams "${N_HPARAMS}" \
  --n_trials "${N_TRIALS}" \
  --command_launcher "multi_gpu" \
  --single_train_envs \
  --skip_confirmation \
  --hparams "${EXTRA_HP}" \
  "${AUG_FLAGS[@]}"

echo
echo "Launched sweep:"
echo "  Dataset: ${DATASET}"
echo "  Algorithm: ${ALGO}"
echo "  Augs: Original + ${AUG2}"
echo "  Trials: ${N_TRIALS}"
echo "  Output: ${OUTPUT_DIR}/${TAG}"