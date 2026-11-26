# BM25 grounding verifier

The raw-QA baseline failed the first real development audit; see
[the audit report](../docs/grounding_audit.md) before bulk scoring. The revised
short-premise candidate is experimental and is not enabled in the production scorer.

This initial verifier exports one `grounding_score` per `node_id` for Tavish's
RL features. It uses the cached question, draft answer, and the exact ordered
context passages. It never reads gold answers, correctness, or `has_answer`.
Sufficiency is not computed: support for a particular draft and whether context
can answer a question are different judgments.

## Method and limitations

The default is `cross-encoder/nli-deberta-v3-small`, a three-label NLI classifier.
See its [model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small).
The hypothesis is the fixed string `Question: ...\nAnswer: ...`; this is a
QA-to-NLI heuristic, not a trained QA-to-declarative converter. Evidence is
processed in overlapping tokenizer windows, repeating the full hypothesis.
The score is the maximum entailment probability over those windows. The model
revision and method parameters are stored in every output row.

Scores are an exploratory, uncalibrated support signal, not a guarantee of
grounding or correctness. A maximum can overrate partially supported answers;
long-distance multi-hop evidence can span windows; contradiction scores are
diagnostics, not a complete conflict-resolution method. Real labeled development
examples must establish whether the signal is useful before reporting verifier
accuracy or selecting a threshold. The ten synthetic fixtures are development
cases, not a benchmark, and the unit tests use a deterministic fake backend.

## Setup and run

From the repository root, use an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-verifier.txt
.venv/bin/hf auth login
.venv/bin/hf download cost-aware-rag/nq-tree15-llama31 --repo-type dataset --local-dir outputs/tree
```

Start with a development pilot, not test-set tuning:

```sh
.venv/bin/python -m verifier.score_nodes \
  --tree outputs/tree/nq_ddev_dpr_tree15_Llama-3.1-8B-Instruct_accumulate.jsonl \
  --passages outputs/tree/passages_final.jsonl \
  --out outputs/verifier/bm25_dev_pilot.jsonl --limit 100
```

Remove `--limit` and use a new output path for the full split. Run train/dev/test
separately; freeze the method and any decision thresholds on train/validation
before evaluating test. Use `--resume` to reuse checked output or its `.partial`
checkpoint; changed context/model configuration is rejected. Missing passages or
model failures stop the command rather than becoming fabricated zero scores.
For legacy nodes lacking `context_docids`, pass `--context-mode accumulate` or
`latest` only after checking generator metadata. Do not silently infer the mode.

Root nodes have `grounding_score: 0`, `status: no_evidence`, `has_evidence: false`.
That says there is no retrieved support, not that the answer is incorrect or
the policy must abstain. Empty answers have a null score and explicit status.
Additional metadata and operation costs accompany Tavish's two required fields:

```json
{"node_id": "qid/R", "grounding_score": 0.87}
```

The actual JSONL also includes qid, status, input/answer fingerprints, context
IDs, verifier configuration, contradiction diagnostics and timing/token costs.
Keep BM25 and e5 files separate: they can share node IDs but differ in context.
The BM25 passage export is complete; e5 requires Chester's separate passage file.

## RL integration and review follow-ups

`verifier.integration.load_scores(path, nodes)` rejects duplicate, missing,
unknown, stale-answer or malformed score rows. It does not recheck context text
because RL's reduced nodes do not include that text: preserve the dataset and
retriever identity of the scoring run. `grounded_node_features(nodes, id, scores)`
appends score, availability, and evidence-presence indicators to the existing
17 features. This is opt-in; published checkpoints and baseline runs are unchanged.
Fit a new scaler/model on train when experimenting with the 20-feature vector.
The existing training entry points do not automatically enable these new features.

The scorer records verifier input tokens, forward calls, and measured latency.
Existing `TreeEnv` rewards do not yet charge verifier execution. Any verifier-on
comparison must add a consistently defined verifier cost for each visited scored
node or report this extra cost separately. Do not compare a free verifier against
costed baselines and claim an improved cost tradeoff.

PR #1 follow-ups: verifier explanation/prompt metadata are optional; the README
development section is near Getting Started; the likelihood/entropy distinction
is explicit; native nodes replace the proposed second cache envelope. The node
adapter sums intermediate drafts and rewrites like `TreeEnv.episode_info` rather
than summing Chester's cumulative fields. Cached latency omits retrieval time and
is batch-amortized, so use call/token counts for primary cost comparisons.

PR #3 review: `rl/features.py` computes inference-time features and reuses
`common.schemas.Action` through `rl.env`. The loader retains gold for rewards and
diagnostics, so passing an entire raw node to a new policy is not safe by default.
Tests verify that changing gold/metrics/has_answer cannot change the existing
feature vector or verifier inputs. Strict whole-word InAcc remains unchanged.
The existing entropy feature groups all omitted vocabulary mass into one bucket;
it is an approximation, not exact vocabulary entropy. Root generation cost is
included in episode call/token reports but is a common omitted reward constant.

Run all dependency-free checks:

```sh
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```

Tavish offered to add CI for these tests; coordinate that workflow with him.


## Reproduce development auditing

Prepare review inputs without gold labels or model scores:

```sh
python3 -m verifier.audit prepare --tree <BM25-ddev-tree> \
  --passages outputs/tree/passages_final.jsonl --out outputs/verifier/audit_new
```

Read every passage for each selected draft, then save separate annotations with
`node_id`, `label` (`supported`, `unsupported`, `unclear`), `evidence_ids`,
`rationale`, `annotator`, and `human_validated`. Record who actually reviewed them;
assistant-generated labels must not be presented as human ground truth. Freeze
annotations before reading scores. Ambiguous cases are excluded from binary
metrics and listed explicitly. Compare the score file using:

```sh
python3 -m verifier.audit evaluate --inputs <review_inputs.jsonl> \
  --annotations <annotations.jsonl> --scores <scores.jsonl>
```

The default 0.5 cutoff is a diagnostic, not a tuned or calibrated policy
threshold. AUC describes ranking on that small audit only. For a fresh audit,
use a different seed and `--exclude-inputs <prior-review-inputs>` so questions do
not overlap. Do not call the fresh development audit a final test set.

`python3 -m verifier.experiment --inputs <review_inputs.jsonl> --out <new-scores.jsonl>
--cache-dir <model-cache>` is the research-only QA2D/short-premise experiment.
It uses [QA2D-t5-small](https://huggingface.co/domenicrosati/QA2D-t5-small) with
lowercase input, punctuation replaced by spaces, a `. ` separator and deterministic
four-beam decoding. It evaluates all single sentences and adjacent sentence pairs
within each passage, never selecting evidence using gold labels. If conversion
drops answer content tokens, it falls back to the raw QA hypothesis and records
that fact. Lexical preservation is only a basic guard: it does not establish
semantic equivalence, preserve numeric ranges reliably, or detect all dropped
question constraints. Inspect the exported hypotheses. The experimental score
file includes conversion operation counts/tokens/latency in addition to NLI
costs. Frozen model revisions and all method settings are recorded per row.


## Guarded claim experiment and human review

See [the follow-up audit](../docs/grounding_audit_followup.md). To reproduce the
v2 experiment, add `--claim-mode guarded-v2` to the experiment command. It keeps
literal ranges, uses grammar templates where recognized, and otherwise rejects
conversions with missing answers or material question constraints. Rejections
have null scores and explicit reasons. `verifier.audit evaluate` reports score
coverage separately; unavailable claims are excluded from AUC and confusion
counts, not converted to negative scores. Do not compare conditional metrics
without also reporting that coverage.

A reviewer packet can be prepared with:

```sh
python3 -m verifier.review_packet --inputs <review_inputs.jsonl> --out <new-packet-dir>
```

This writes a score-blind Markdown packet, whitelisted inputs and a blank JSONL
annotation template. It sends nothing. Human validation remains a real external
review task; an assistant-created packet or label file is not human review.
