ALGORITHM="ERM_GRAD_INHIB_FAST_conditional_normalized2_dp_thresh_layer"
DATASET="OfficeHome"
SWEEP="sweep2"

# make output dir if it does not exist
mkdir -p /home/ahmedm04/projects/CausalGradient/results/${DATASET}/ablations/${ALGORITHM}/${SWEEP}

python -m domainbed.scripts.sweep launch\
       --data_dir=/home/ahmedm04/dg_datasets\
       --output_dir=/home/ahmedm04/projects/CausalGradient/results/${DATASET}/ablations/${ALGORITHM}/${SWEEP}\
       --command_launcher multi_gpu\
       --algorithms ${ALGORITHM}\
       --datasets ${DATASET}\
       --n_hparams 30\
       --n_trials 1\
       --skip_confirmation\
       --single_test_envs


python -m domainbed.scripts.collect_results_by_all_hparams\
    --input_dir /home/ahmedm04/projects/CausalGradient/results/${DATASET}/ablations/${ALGORITHM}/${SWEEP}\
    --algorithm ${ALGORITHM}