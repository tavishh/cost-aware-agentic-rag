#!/bin/bash
#SBATCH --partition=rtx-devel
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tree15_mh_%j.out
# Multi-hop version of run_tree15.sh (BM25, Llama-3.1-8B, k=5, accumulate); only the corpus differs.
# Usage (from the project root):
#   MH_INDEX_DIR=<dir with lucene-{hotpotqa,2wikimultihopqa,musique}> DS=musique SPLITS="test ddev dtrain" \
#     sbatch -A <slurm account> run_tree15_mh.sh
# (or set BM25_INDEX directly to one index)
set -e
: "${DS:?set DS to hotpotqa, 2wikimultihopqa or musique}"
if [ -z "$BM25_INDEX" ]; then
  : "${MH_INDEX_DIR:?set MH_INDEX_DIR (or BM25_INDEX)}"
  BM25_INDEX=$MH_INDEX_DIR/lucene-$DS
fi
export BM25_INDEX
module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"

SHORT=$DS; [ "$DS" = "2wikimultihopqa" ] && SHORT=2wiki
MODEL=meta-llama/Llama-3.1-8B-Instruct
declare -A N=([test]=500 [ddev]=500 [dtrain]=4000)

for SPLIT in ${SPLITS:-test}; do
  echo "===== $SHORT $SPLIT (BM25: $BM25_INDEX) ====="
  C=outputs/retrieval/${SHORT}_${SPLIT}_bm25_top50.jsonl
  A="--model $MODEL --cache $C --n ${N[$SPLIT]}"

  conda activate pyserini
  python cache/build_bm25_cache.py --questions outputs/retrieval/${SHORT}_${SPLIT}_questions.jsonl --cache $C --n ${N[$SPLIT]}

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
