#!/bin/bash
#SBATCH --job-name=e5dump
#SBATCH --partition=rtx-devel
#SBATCH --account=p2026_0112_neu
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module load miniforge3/25.3.0-3; eval "$(conda shell.bash hook)"; conda activate pyserini
IDX=$HOME/.cache/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.7b58c08da992b2ea7e96667f0b176651
cd ~/cost-aware-agentic-rag/retrieval
python e5_dump_corpus.py --index $IDX --outdir /scratch/li_z33_neu/e5/corpus --num_shards 8
