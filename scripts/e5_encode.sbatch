#!/bin/bash
#SBATCH --job-name=e5enc
#SBATCH --partition=rtx-devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
# Encode the 8 corpus shards with e5-base-v2 (~19 min per shard on an RTX PRO 6000).
# Usage (from the project root):
#   E5_CORPUS_DIR=<dump dir> E5_EMB_DIR=<out dir> [E5_MODEL_DIR=<local model>] \
#     sbatch -A <slurm account> scripts/e5_encode.sbatch
set -e
: "${E5_CORPUS_DIR:?set E5_CORPUS_DIR}" "${E5_EMB_DIR:?set E5_EMB_DIR}"
module load miniforge3/25.3.0-3; eval "$(conda shell.bash hook)"; conda activate vllm
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p "$E5_EMB_DIR"
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}/retrieval"
python e5_encode.py --shard $SLURM_ARRAY_TASK_ID --corpus_dir "$E5_CORPUS_DIR" --out_dir "$E5_EMB_DIR"
