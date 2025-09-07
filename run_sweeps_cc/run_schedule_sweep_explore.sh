set -euo pipefail

# Determine base directory (parent of run_sweeps)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

# Required environment variables (must be set by the job script)
if [[ -z "${ALGORITHM:-}" || -z "${DATASET:-}" || -z "${SWEEP:-}" ]]; then
  echo "ERROR: Please export ALGORITHM, DATASET, and SWEEP before running." >&2
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

OUT_DIR="${BASE_DIR}/results/${DATASET}/schedule_explore/${ALGORITHM}/${SWEEP}"
mkdir -p "$OUT_DIR"

echo "Using data_dir: $DATA_SRC"
echo "Output to:     $OUT_DIR"
echo "Dataset:       $DATASET"
echo "Algorithm:     $ALGORITHM"

python -m domainbed.scripts.sweep_schedule launch \
  --data_dir="${DATA_SRC}" \
  --output_dir="${OUT_DIR}" \
  --command_launcher multi_gpu \
  --algorithms ${ALGORITHM} \
  --datasets ${DATASET} \
  --n_hparams 30 \
  --n_trials 1 \
  --skip_confirmation \
  --single_test_envs

python -m domainbed.scripts.collect_results_by_all_hparams \
  --input_dir "${OUT_DIR}" \
  --algorithm ${ALGORITHM}
