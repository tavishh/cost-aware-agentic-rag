# Role 3 shared contract

Status: shared contracts with a native-node adapter and initial offline verifier. This defines interfaces and
test data. The initial NLI grounding implementation is documented in
`verifier/README.md`; existing RL and threshold policies live in `rl/`.
Python 3.9+ and the standard library suffice for the contract tests.

## Scope and semantics

The proposal specifies a frozen generator, answer-token log probabilities,
five passages per retrieval, and up to three retrieval/rewrite steps. Budget
defaults reflect that proposal; training code remains responsible for enforcing
the selected experiment budget. REWRITE includes query rewriting followed by
retrieval and consumes one step. ANSWER and ABSTAIN terminate the episode.

`common/interfaces.py` defines structural adapter interfaces, backed by frozen
dataclasses in `common/schemas.py`. All adapters must construct those types;
the types are not a general parser for arbitrary untrusted JSON.

| Interface | Input | Output | Integration owner |
| --- | --- | --- | --- |
| `Generator.generate` | question, ordered passages | `GenerationResult` | Role 1/shared generator |
| `UncertaintyEstimator.estimate` | generation | `UncertaintyResult` | Role 3 |
| `Verifier.check_sufficiency` | question, ordered passages | `VerificationResult` | Role 3 |
| `Verifier.verify_answer` | question, draft, ordered passages | `VerificationResult` | Role 3 |
| `Policy.select_action` | `Observation` | `Action` | Role 3 thresholds / Role 4 RL |

Sufficiency asks whether the context can answer the question. Grounding asks
whether the context supports every factual assertion in the particular draft.
A sufficient context can accompany an unsupported draft. Neither is world-truth
correctness. NO includes absent support; it does not necessarily mean contradiction.
UNKNOWN represents ambiguity, conflicting evidence, or a failed judgment, with
an explanation. No evidence yields NO for a factual draft in the fixture.
An uncomputed verifier result is `None`, distinct from an attempted UNKNOWN.

Optional verifier scores must be in [0, 1], higher meaning more support. They
are not calibrated probabilities unless separately validated. UNKNOWN has no
score. The verifier version must identify model and judgment/aggregation method;
the optional prompt version identifies its exact prompt for prompted models.
NLI models need not supply a prompt version or a natural-language explanation. Evidence IDs must reference
passages actually supplied. Claim-level diagnostics can be added in a future
schema version when the verifier implementation selects an aggregation method.

Uncertainty increases with the numeric value. The first planned method is
`mean_nll_v1 = -mean(answer_token_logprobs)`, with natural logarithms and no
prompt, padding, or special tokens. This is not token entropy. Missing log
probabilities produce `value=None` with a reason, never zero. BM25 relevance
scores remain separate. Thresholds are method/model specific and will be fit
on development data, not the test split.

## Labels and cache boundary

The existing retrieval JSONL contains `qid`, `question`, `gold`, `docids`,
`scores`, and `has_answer`. It stays unchanged. An adapter should resolve each
docid using `get_text`, combine it with its score into a `Passage`, generate a
draft, compute signals, and explicitly construct an `Observation`. Do not pass
the retrieval row wholesale to a policy. `has_answer` uses gold-answer substring
matching and is solely offline retrieval metadata.

The canonical state cache is Chester's existing tree JSONL. Do not introduce a
second envelope. `common/node_adapter.py` converts raw nodes to observations,
resolves exact `context_docids`, maps field names and seconds to milliseconds,
and follows `parent_id` to accumulate each visited draft and rewrite once.
Gold, metrics and `has_answer` are never copied to the observation. The adapter
preserves repeated passage occurrences because the generator saw them too.

`node_id` identifies a node within a versioned dataset/retriever run. BM25 and
e5 reuse IDs, so keep their score files separate. Model/prompt/cache revisions
remain part of experiment identity, not a replacement for native node IDs.
The adapter's `greedy_top1_mean_nll_v1` is an explicit proxy over cached positions
when greedy decoding is confirmed. It does not claim to reconstruct cleaned
answer token alignment. Selected-token logprobs remain None in that adapter.

`Cost` on each result records that operation's calls/tokens/latency. Observation
`cumulative_cost` records all work so far, including retrieval, query rewriting,
generation and both verifier checks. Count each backend invocation once; if one
call returns both checks, account for it only once in the cumulative total.
Rewrite LLM calls count as generation calls. Cache replay must charge original
operation costs, not lookup latency. The cache/environment owns accumulation;
these dataclasses do not infer costs or budgets.

## Fixture and validation

`tests/fixtures/verifier_cases.jsonl` contains ten synthetic, author-labeled
examples: direct support, contradiction, incidental answer mention, paraphrase,
partial support, complete and broken multi-hop chains, empty evidence,
irrelevant evidence, and unresolved conflicting sources. Expected labels are
under `expected`; only `input` should be passed to the future verifier.
This is a development fixture, not human-validated benchmark data or a source
for choosing test thresholds. Correct grounding requires all factual claims.

Run from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

These tests validate contracts and fixture integrity, not model accuracy. They
avoid importing `retrieval.bm25`, which loads a Lucene index at import time.

## Next handoffs

1. Role 1: expose generation and log probabilities through the generator
   interface, resolve passage text, and version the future state cache.
2. Role 3: implement uncertainty extraction, separate sufficiency/grounding
   checks, failure handling and threshold policies against these contracts.
3. Role 2: label an independent verifier evaluation subset, define metric
   denominators and compare policies with matched budgets and verifier costs.
4. Role 4: consume inference observations, enforce transitions/budgets, and
   isolate reward/oracle labels from policy features.

Reference: [Sufficient Context](https://arxiv.org/abs/2411.06037) motivates the
context-sufficiency distinction; these interfaces are project design choices,
not a claim to reproduce that paper's method.
