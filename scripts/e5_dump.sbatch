#!/bin/bash
#SBATCH --job-name=e5dump
#SBATCH --partition=rtx-devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Dump the DPR wiki passages from the local Lucene index (CPU work; rtx-devel requires a GPU).
# Usage (from the project root):
#   DPR_INDEX=<path to Pyserini wikipedia-dpr-100w index> E5_CORPUS_DIR=<out dir> \
#     sbatch -A <slurm account> scripts/e5_dump.sbatch
set -e
: "${DPR_INDEX:?set DPR_INDEX}" "${E5_CORPUS_DIR:?set E5_CORPUS_DIR}"
module load miniforge3/25.3.0-3; eval "$(conda shell.bash hook)"; conda activate pyserini
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}/retrieval"
python e5_dump_corpus.py --index "$DPR_INDEX" --outdir "$E5_CORPUS_DIR" --num_shards 8
