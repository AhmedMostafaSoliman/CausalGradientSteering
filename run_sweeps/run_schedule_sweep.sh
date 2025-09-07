#!/bin/bash

# Activate the specified environment
source /home/ahmedm04/projects/PEFTFusion/.venv/bin/activate

# Define hyperparameters and config
ALGORITHM="ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule"
DATASET="PACS"   # 4 domains; sweep will iterate test envs
SHUFFLE_PATCH_SIZE=2   # initial; train_schedule will move 1 -> 2 -> 4
SHUFFLE_PATCH_EVERY=200
SHUFFLE_FN="shuffle_center_patches"
GRAD_INHIB_STRENGTH=0.4
DOT_PRODUCT_THRESH=0.4

# Other hparams
BATCH_SIZE=40
WEIGHT_DECAY=0.0009 #0.0005
LR=1e-5 #0.003
RESNET_DROPOUT=0.25

OUTPUT_DIR="/home/ahmedm04/projects/CausalGradient/results/${DATASET}/schedule_sweep/${ALGORITHM}/init_ps=${SHUFFLE_PATCH_SIZE}__every=${SHUFFLE_PATCH_EVERY}__fn=${SHUFFLE_FN}__gamma=${GRAD_INHIB_STRENGTH}__dotth=${DOT_PRODUCT_THRESH}__bs=${BATCH_SIZE}__wd=${WEIGHT_DECAY}__lr=${LR}__dropout=${RESNET_DROPOUT}"
mkdir -p "${OUTPUT_DIR}"

python -m domainbed.scripts.sweep_schedule \
  launch \
  --datasets ${DATASET} \
  --algorithms ${ALGORITHM} \
  --n_hparams_from 0 \
  --n_hparams 1 \
  --n_trials 1 \
  --output_dir ${OUTPUT_DIR} \
  --data_dir /home/ahmedm04/dg_datasets \
  --command_launcher multi_gpu \
  --hparams "{\"shuffle_patch_size\": ${SHUFFLE_PATCH_SIZE}, \"shuffle_patch_every\": ${SHUFFLE_PATCH_EVERY}, \"shuffle_fn\": \"${SHUFFLE_FN}\", \"grad_inhib_strength\": ${GRAD_INHIB_STRENGTH},\"dot_prod_thresh\": ${DOT_PRODUCT_THRESH}, \"batch_size\": ${BATCH_SIZE}, \"weight_decay\": ${WEIGHT_DECAY}, \"lr\": ${LR}, \"resnet_dropout\": ${RESNET_DROPOUT}}" \
  --single_test_envs \
  --skip_confirmation

python -m domainbed.scripts.collect_results --input_dir "${OUTPUT_DIR}"

