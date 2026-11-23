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

## e5 and multi-hop pipelines

All job scripts run from the project root. Machine-specific paths come from environment
variables (no defaults that point at a personal directory); pass the SLURM account on the
command line with `sbatch -A <account>`. `PROJECT_DIR` defaults to the submit directory.

| Variable | Used by | Meaning |
|---|---|---|
| `DPR_INDEX` | `scripts/e5_dump.sbatch` | Pyserini `wikipedia-dpr-100w` Lucene index (local path) |
| `E5_CORPUS_DIR` | `scripts/e5_*.sbatch` | dumped corpus shards (`corpus_XX.jsonl`) |
| `E5_EMB_DIR` | `scripts/e5_encode.sbatch`, `run_tree15_e5.sh`, `cache/e5_retrieve.py`, `retrieval/e5_search.py --emb_dir` | e5 embeddings (`emb_XX.npy`, `ids_XX.npy`, 8 shards, ~31 GB fp16) |
| `E5_MODEL_DIR` | `retrieval/e5_encode.py` (also imported by `e5_search.py`), `cache/e5_retrieve.py` | e5-base-v2: local copy or HF id (default `intfloat/e5-base-v2`) |
| `BM25_INDEX` | `retrieval/bm25.py` | local Lucene index; unset = DPR wiki (NQ), so the NQ pipeline is unchanged |
| `MH_INDEX_DIR` | `run_tree15_mh.sh` | directory with `lucene-{hotpotqa,2wikimultihopqa,musique}` |

### e5 (NQ, second retriever setting)

1. Dump the DPR corpus from the Lucene index (docids stay aligned with BM25):
   `DPR_INDEX=... E5_CORPUS_DIR=... sbatch -A <acct> scripts/e5_dump.sbatch`
2. Encode the 21M passages (8-shard job array):
   `E5_CORPUS_DIR=... E5_EMB_DIR=... sbatch -A <acct> scripts/e5_encode.sbatch`
3. Question files for the three NQ splits (Mac/local, from existing caches/trees):
   `python -m retrieval.make_questions --inp outputs/retrieval/nq_arag_test_bm25_top50.jsonl --out outputs/retrieval/nq_arag_test_questions.jsonl`
   (same for `nq_ddev_dpr` / `nq_dtrain_dpr` from their tree files)
4. Trees, one split per job:
   `E5_EMB_DIR=... SPLITS=arag_test sbatch -A <acct> run_tree15_e5.sh` (then `ddev_dpr`, `dtrain_dpr`)
5. Summaries: `python cache/tree_summary.py --select <ddev tree> <test tree> <ddev tree> <dtrain tree>`

### Multi-hop (HotpotQA, 2WikiMultiHopQA, MuSiQue; BM25)

Raw data: Adaptive-RAG `processed_data/` (test sets) and `raw_data/` (HotpotQA questions from
`hotpotqa/hotpot_qa` parquet on HF, HotpotQA corpus from `mteb/hotpotqa` corpus.jsonl since the
original dumps are offline; 2Wiki and MuSiQue from the IRCoT download links).

1. Corpora (IRCoT `build_index.py` logic, docids prefixed per dataset), run where `raw_data/` lives:
   `python retrieval/build_mh_corpus.py {hotpotqa|2wikimultihopqa|musique} --out <corpus dir>`
2. Lucene indexes:
   `python -m pyserini.index.lucene --collection JsonCollection --input <corpus dir>/<ds> --index <MH_INDEX_DIR>/lucene-<ds> --generator DefaultLuceneDocumentGenerator --threads 8 --storeRaw`
3. Question files (test = Adaptive-RAG's 500; dtrain 4000 / ddev 500 from the original dev set for
   HotpotQA/2Wiki and from train for MuSiQue; single reference answer, no aliases):
   `python retrieval/build_mh_questions.py <ds> --out_dir outputs/retrieval`
4. Trees: `MH_INDEX_DIR=... DS=<ds> SPLITS="test ddev dtrain" sbatch -A <acct> run_tree15_mh.sh`
   (`cache/build_bm25_cache.py` builds the top-50 cache and `passages_L0` inside the job)
5. Summaries: `python cache/tree_summary.py --select <ds ddev tree> <ds test/ddev/dtrain trees>`
6. Passage files for the verifier:
   `python cache/build_passages_final.py --out outputs/tree/passages_final_<name>.jsonl <tree files>`

Published outputs: NQ BM25/e5 trees and passages in `cost-aware-rag/nq-tree15-llama31`; multi-hop
trees and passages in `cost-aware-rag/mh-tree15-llama31`; full tables in
`results/trees/tree15_summaries_strict.txt`.

### Script status

Current pipeline:
`cache/e5_retrieve.py`, `cache/build_bm25_cache.py`, `cache/tree_summary.py`,
`cache/build_passages_final.py`, `retrieval/bm25.py`, `retrieval/e5_dump_corpus.py`,
`retrieval/e5_encode.py`, `retrieval/build_mh_corpus.py`, `retrieval/build_mh_questions.py`,
`retrieval/make_questions.py`, `run_tree15_e5.sh`, `run_tree15_mh.sh`, `scripts/e5_*.sbatch`.

Kept for the record (one-off analyses behind reported numbers; not needed to rebuild trees):
- `retrieval/e5_search.py` + `retrieval/e5_add_has_answer.py`: standalone e5 top-50 search for an
  arbitrary question file, used for the BM25-vs-e5 Hit@k comparison and the contamination check
  (the tree pipeline uses `cache/e5_retrieve.py` instead).
- `retrieval/compare_hitk.py`: Hit@k for BM25 / e5 / RRF (RRF was dropped).
- `retrieval/make_dpr_nq_test.py` + `retrieval/contamination_check.py`: contamination check on
  DPR NQ test and the rest of DPR NQ dev.
- `cache/compare_settings.py`: compares answer prompting/decoding settings (simple prompt with newline stop, no stop, Adaptive-RAG few-shot) on the root and R nodes of NQ test; part of the gap analysis vs. Moskvoretskii et al. that ruled out prompt and decoding differences.
