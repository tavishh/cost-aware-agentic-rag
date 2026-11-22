#!/bin/bash
#SBATCH --job-name=e5enc
#SBATCH --partition=rtx-devel
#SBATCH --account=p2026_0112_neu
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
module load miniforge3/25.3.0-3; eval "$(conda shell.bash hook)"; conda activate vllm
export HF_HOME=/scratch/li_z33_neu/hf_cache; export HF_HUB_OFFLINE=1
mkdir -p /scratch/li_z33_neu/e5/emb
cd ~/cost-aware-agentic-rag/retrieval
python e5_encode.py --shard $SLURM_ARRAY_TASK_ID --corpus_dir /scratch/li_z33_neu/e5/corpus --out_dir /scratch/li_z33_neu/e5/emb
