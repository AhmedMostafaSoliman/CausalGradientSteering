ALGORITHM="ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule"
DATASET="VLCS"
SWEEP="sweep1"

# Determine base directory (parent of run_sweeps)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

# make output dir if it does not exist
mkdir -p "${BASE_DIR}/results/${DATASET}/schedule_explore/${ALGORITHM}/${SWEEP}"

python -m domainbed.scripts.sweep_schedule launch\
       --data_dir=/home/ahmedm04/dg_datasets\
       --output_dir="${BASE_DIR}/results/${DATASET}/schedule_explore/${ALGORITHM}/${SWEEP}"\
       --command_launcher multi_gpu\
       --algorithms ${ALGORITHM}\
       --datasets ${DATASET}\
       --n_hparams 20\
       --n_trials 1\
       --skip_confirmation\
       --single_test_envs


python -m domainbed.scripts.collect_results_by_all_hparams\
    --input_dir "results/${DATASET}/schedule_explore/${ALGORITHM}/${SWEEP}"\
    --algorithm ${ALGORITHM}
