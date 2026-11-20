# Role 1: building the cached environment

All scripts run from the repo root. Retrieval steps need Pyserini (BM25, `wikipedia-dpr-100w`); tree steps need a GPU with vLLM.

## Downloads

- **Adaptive-RAG test set:** `processed_data.tar.gz` from https://github.com/starsuzi/Adaptive-RAG, then extract `processed_data/nq/test_subsampled.jsonl`
- **DPR NQ dev** (used to filter train/dev): https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz, saved to `data/raw/` (or set `DPR_NQ_DEV`)

## Run order

```bash
# 1. Test cache (Adaptive-RAG 500 questions)
python cache/build_retrieval_cache_arag.py --src <path>/test_subsampled.jsonl

# 2. Train/dev candidates from NQ dev, excluding test
python cache/build_retrieval_cache_nqdev.py

# 3. Keep only DPR NQ dev questions and top up to 4,000 / 500
python cache/check_dpr_subset.py
python cache/build_dpr_topup.py
python cache/export_passages.py

# 4. Build 15-node trees (GPU)
SPLITS="arag_test dtrain ddev dtraintop ddevtop" bash run_tree15.sh
# on Slurm: sbatch --account=<acct> --export=ALL,SPLITS="..."
