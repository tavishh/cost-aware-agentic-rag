#!/bin/bash
#SBATCH --partition=rtx-devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tree15_e5_%j.out
# e5 version of run_tree15.sh: same settings, only the retriever differs.
# Usage (from the project root):
#   E5_EMB_DIR=<dir with emb_XX.npy> [E5_MODEL_DIR=<local model>] SPLITS=arag_test \
#     sbatch -A <slurm account> run_tree15_e5.sh
set -e
: "${E5_EMB_DIR:?set E5_EMB_DIR}"
export E5_EMB_DIR E5_MODEL_DIR
module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"

MODEL=meta-llama/Llama-3.1-8B-Instruct
declare -A N=([arag_test]=500 [dtrain_dpr]=4000 [ddev_dpr]=500)

for SPLIT in ${SPLITS:-arag_test}; do
  echo "===== $SPLIT (e5) ====="
  C=outputs/retrieval/nq_${SPLIT}_e5_top50.jsonl
  A="--model $MODEL --cache $C --n ${N[$SPLIT]}"

  conda activate vllm
  python cache/e5_retrieve.py search-cache $A --questions outputs/retrieval/nq_${SPLIT}_questions.jsonl
  conda activate pyserini
  python cache/e5_retrieve.py annotate-cache $A

  for L in 0 1 2; do
    conda activate vllm
    python cache/tree_expand.py $A --layer $L
    python cache/e5_retrieve.py search-layer $A --layer $((L+1))
    conda activate pyserini
    python cache/e5_retrieve.py annotate-layer $A --layer $((L+1))
  done
  conda activate vllm
  python cache/tree_expand.py $A --layer 3
  python cache/tree_answer.py $A
done
