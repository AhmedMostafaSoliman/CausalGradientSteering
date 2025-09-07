#!/bin/bash
# filepath: /home/ahmedm04/projects/DomainBed/run_specific_hparam.sh

# Define your hyperparameters as variables
ALGORITHM="ERM_GRAD_INHIB_FAST_dp_thresh_layer_selinhib"
DATASET="OfficeHome"
SHUFFLE_PATCH_SIZE=2
SHUFFLE_PATCH_EVERY=700
SHUFFLE_FN="shuffle_center_patches"
GRAD_INHIB_STRENGTH=0.7
# other hparams
BATCH_SIZE=32
WEIGHT_DECAY=0.002
LR=1e-5
RESNET_DROPOUT=0

# Create output directory name with hyperparameter values
OUTPUT_DIR="/home/ahmedm04/projects/CausalGradient/results/${DATASET}/final/${ALGORITHM}/shuffle_patch_size=${SHUFFLE_PATCH_SIZE}__shuffle_patch_every=${SHUFFLE_PATCH_EVERY}__shuffle_fn=${SHUFFLE_FN}__grad_inhib_strength=${GRAD_INHIB_STRENGTH}__batch_size=${BATCH_SIZE}__weight_decay=${WEIGHT_DECAY}__lr=${LR}__resnet_dropout=${RESNET_DROPOUT}"

# Create the output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Run the sweep command with the variables
python -m domainbed.scripts.sweep \
    launch \
    --datasets ${DATASET} \
    --algorithms ${ALGORITHM} \
    --n_hparams_from 0 \
    --n_hparams 1 \
    --n_trials 1 \
    --output_dir ${OUTPUT_DIR} \
    --data_dir=/home/ahmedm04/dg_datasets\
    --command_launcher multi_gpu \
    --hparams "{\"shuffle_patch_size\": ${SHUFFLE_PATCH_SIZE}, \"shuffle_patch_every\": ${SHUFFLE_PATCH_EVERY}, \"shuffle_fn\": \"${SHUFFLE_FN}\", \"grad_inhib_strength\": ${GRAD_INHIB_STRENGTH}, \"batch_size\":${BATCH_SIZE}, \"weight_decay\":${WEIGHT_DECAY}, \"lr\":${LR}, \"resnet_dropout\":${RESNET_DROPOUT}}" \
    --single_test_env \
    --skip_confirmation


python -m domainbed.scripts.collect_results --input_dir "${OUTPUT_DIR}"