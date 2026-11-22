#!/bin/bash
#SBATCH --partition=rtx-devel
#SBATCH --account=p2026_0112_neu
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tree15_e5_%j.out
# Usage: SPLITS=arag_test sbatch run_tree15_e5.sh   (one split per job is safest)
# Same settings as run_tree15.sh; only the retriever differs (e5 instead of BM25).

set -e
module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export E5_EMB_DIR=${E5_EMB_DIR:-/work/neu/p2026_0112_neu/e5/emb}
export E5_MODEL_DIR=${E5_MODEL_DIR:-$HOME/models/e5-base-v2}
cd ~/cost-aware-agentic-rag

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
