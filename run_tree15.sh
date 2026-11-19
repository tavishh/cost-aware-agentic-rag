#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --account=p2026_0112_neu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/tree15_%j.out

set -e
module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
export HF_HOME=/scratch/li_z33_neu/hf_cache
export HF_HUB_OFFLINE=1
cd ~/cost-aware-agentic-rag

MODEL=meta-llama/Llama-3.1-8B-Instruct
declare -A N=([test]=500 [train]=800 [dev]=200 [arag_test]=500 [dtrain]=4000 [ddev]=500 [dtraintop]=1137 [ddevtop]=141)

for SPLIT in ${SPLITS:-test train dev}; do
  echo "===== $SPLIT ====="
  A="--model $MODEL --cache outputs/retrieval/nq_${SPLIT}_bm25_top50.jsonl --n ${N[$SPLIT]}"
  for L in 0 1 2; do
    conda activate vllm
    python cache/tree_expand.py $A --layer $L
    conda activate pyserini
    python cache/rewrite_retrieve.py $A --layer $((L+1))
  done
  conda activate vllm
  python cache/tree_expand.py $A --layer 3
  python cache/tree_answer.py $A
done
