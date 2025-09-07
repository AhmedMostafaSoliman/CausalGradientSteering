#!/bin/bash

# Activate the specified environment
source /home/ahmedm04/projects/PEFTFusion/.venv/bin/activate

# Grid search parameters - define arrays for each parameter to sweep
ALGORITHM="ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule"
DATASET="PACS"
SHUFFLE_PATCH_SIZES=(1)  # initial (will be scheduled to 2 then 4)
SHUFFLE_PATCH_EVERYS=(100 125 150)
SHUFFLE_FNS=("shuffle_center_patches")
GRAD_INHIB_STRENGTHS=(0.4 0.5 0.6)
BATCH_SIZES=(8 16 32)
WEIGHT_DECAYS=(0.001 0.002 0.003)
LRS=(1e-5)
RESNET_DROPOUTS=(0 0.25 0.5)

# Grid search loops
for SHUFFLE_PATCH_SIZE in "${SHUFFLE_PATCH_SIZES[@]}"; do
  for SHUFFLE_PATCH_EVERY in "${SHUFFLE_PATCH_EVERYS[@]}"; do
    for SHUFFLE_FN in "${SHUFFLE_FNS[@]}"; do
      for GRAD_INHIB_STRENGTH in "${GRAD_INHIB_STRENGTHS[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
          for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
            for LR in "${LRS[@]}"; do
              for RESNET_DROPOUT in "${RESNET_DROPOUTS[@]}"; do
                
                OUTPUT_DIR="/home/ahmedm04/projects/CausalGradient/results/${DATASET}/schedule_sweep/${ALGORITHM}/init_ps=${SHUFFLE_PATCH_SIZE}__every=${SHUFFLE_PATCH_EVERY}__fn=${SHUFFLE_FN}__gamma=${GRAD_INHIB_STRENGTH}__bs=${BATCH_SIZE}__wd=${WEIGHT_DECAY}__lr=${LR}__dropout=${RESNET_DROPOUT}"
                mkdir -p "${OUTPUT_DIR}"

                echo "Running with: ps=${SHUFFLE_PATCH_SIZE}, every=${SHUFFLE_PATCH_EVERY}, fn=${SHUFFLE_FN}, gamma=${GRAD_INHIB_STRENGTH}, bs=${BATCH_SIZE}, wd=${WEIGHT_DECAY}, lr=${LR}, dropout=${RESNET_DROPOUT}"

                # Run the scheduled sweep script
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
                  --hparams "{\"shuffle_patch_size\": ${SHUFFLE_PATCH_SIZE}, \"shuffle_patch_every\": ${SHUFFLE_PATCH_EVERY}, \"shuffle_fn\": \"${SHUFFLE_FN}\", \"grad_inhib_strength\": ${GRAD_INHIB_STRENGTH}, \"batch_size\": ${BATCH_SIZE}, \"weight_decay\": ${WEIGHT_DECAY}, \"lr\": ${LR}, \"resnet_dropout\": ${RESNET_DROPOUT}}" \
                  --single_test_envs \
                  --skip_confirmation
                
                python -m domainbed.scripts.collect_results --input_dir "${OUTPUT_DIR}"
              done
            done
          done
        done
      done
    done
  done
done

