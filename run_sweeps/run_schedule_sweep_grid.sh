#!/bin/bash

# Activate the specified environment
source /home/ahmedm04/projects/PEFTFusion/.venv/bin/activate

# Determine base directory (parent of run_sweeps)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

# Grid search parameters - define arrays for each parameter to sweep
ALGORITHM="ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule"
DATASET="PACS"

# Cosine schedule endpoints (keep small to control grid size)
PS_STARTS=(1)
PS_ENDS=(8)
CROP_STARTS=(1.0)
CROP_ENDS=(0.5)
GAMMA_STARTS=(1.0)
GAMMA_ENDS=(0.2)
SCHED_START_PCTS=(0.0)
SCHED_END_PCTS=(1.0)

# Base hparams grids
SHUFFLE_PATCH_EVERYS=(100 150 200)
SHUFFLE_FNS=("shuffle_center_patches")
DOT_PRODUCT_THRESHS=(0.4)
BATCH_SIZES=(8 16 32)
WEIGHT_DECAYS=(0.001 0.002 0.003)
LRS=(1e-5)
RESNET_DROPOUTS=(0 0.25 0.5)

# Grid search loops
for PS_START in "${PS_STARTS[@]}"; do
  for PS_END in "${PS_ENDS[@]}"; do
    for CROP_START in "${CROP_STARTS[@]}"; do
      for CROP_END in "${CROP_ENDS[@]}"; do
        for GAMMA_START in "${GAMMA_STARTS[@]}"; do
          for GAMMA_END in "${GAMMA_ENDS[@]}"; do
            for SCHED_START_PCT in "${SCHED_START_PCTS[@]}"; do
              for SCHED_END_PCT in "${SCHED_END_PCTS[@]}"; do
                for SHUFFLE_PATCH_EVERY in "${SHUFFLE_PATCH_EVERYS[@]}"; do
                  for SHUFFLE_FN in "${SHUFFLE_FNS[@]}"; do
                    for DOT_PRODUCT_THRESH in "${DOT_PRODUCT_THRESHS[@]}"; do
                      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                        for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                          for LR in "${LRS[@]}"; do
                            for RESNET_DROPOUT in "${RESNET_DROPOUTS[@]}"; do
                
                OUTPUT_DIR="${BASE_DIR}/results/${DATASET}/schedule_sweep/${ALGORITHM}/ps=${PS_START}to${PS_END}__crop=${CROP_START}to${CROP_END}__gamma=${GAMMA_START}to${GAMMA_END}__sched=${SCHED_START_PCT}to${SCHED_END_PCT}__every=${SHUFFLE_PATCH_EVERY}__fn=${SHUFFLE_FN}__dotth=${DOT_PRODUCT_THRESH}__bs=${BATCH_SIZE}__wd=${WEIGHT_DECAY}__lr=${LR}__dropout=${RESNET_DROPOUT}"
                mkdir -p "${OUTPUT_DIR}"

                echo "Running with: ps=${PS_START}->${PS_END}, crop=${CROP_START}->${CROP_END}, gamma=${GAMMA_START}->${GAMMA_END}, sched=${SCHED_START_PCT}->${SCHED_END_PCT}, every=${SHUFFLE_PATCH_EVERY}, fn=${SHUFFLE_FN}, dotth=${DOT_PRODUCT_THRESH}, bs=${BATCH_SIZE}, wd=${WEIGHT_DECAY}, lr=${LR}, dropout=${RESNET_DROPOUT}"

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
                  --hparams "{\"shuffle_patch_size\": ${PS_START}, \"shuffle_patch_every\": ${SHUFFLE_PATCH_EVERY}, \"shuffle_fn\": \"${SHUFFLE_FN}\", \"grad_inhib_strength\": ${GAMMA_START}, \"dot_prod_thresh\": ${DOT_PRODUCT_THRESH}, \"batch_size\": ${BATCH_SIZE}, \"weight_decay\": ${WEIGHT_DECAY}, \"lr\": ${LR}, \"resnet_dropout\": ${RESNET_DROPOUT}, \"shuffle_patch_size_start\": ${PS_START}, \"shuffle_patch_size_end\": ${PS_END}, \"center_crop_ratio_start\": ${CROP_START}, \"center_crop_ratio_end\": ${CROP_END}, \"grad_inhib_strength_start\": ${GAMMA_START}, \"grad_inhib_strength_end\": ${GAMMA_END}, \"sched_start_pct\": ${SCHED_START_PCT}, \"sched_end_pct\": ${SCHED_END_PCT}}" \
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
            done
          done
        done
      done
    done
  done
done
