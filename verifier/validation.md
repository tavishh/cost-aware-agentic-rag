# Initial verifier validation

The public `cross-encoder/nli-deberta-v3-small` model was run on the synthetic
fixture with revision `fa2804872c3b4bd748f38c0185cc85775361e735`, on CPU.
These examples are not a human-validated evaluation set, and no threshold was
tuned from these scores.

| Synthetic case | Expected grounding | Entailment score |
| --- | --- | --- |
| Direct support | yes | 0.9759 |
| Contradiction | no | 0.0010 |
| Incidental answer mention | no | 0.0428 |
| Paraphrase | yes | 0.8992 |
| Unsupported additional claim | no | approximately 0 |
| Two-passage multi-hop support | yes | 0.0105 |
| Missing evidence hop | no | 0.0002 |
| Empty context | no | 0 (no model call) |
| Irrelevant context | no | approximately 0 |
| Conflicting sources | unknown | 0.0263 |

The low score on supported multi-hop evidence is a known failure. Low support on
conflicting sources is not the same as detecting an unknown/ambiguous verdict.
The current scorer intentionally exports a continuous support signal rather
than claiming a calibrated binary or three-way verifier.

The cache adapter was also checked on 200 real BM25 development nodes: draft,
rewrite and retrieval call counts and token totals match `TreeEnv.episode_info`.
All 29 dependency-free contract/pipeline tests pass. Those tests establish
plumbing and invariants, not model accuracy.

Before claiming a useful RL improvement, label a question-stratified sample of
real BM25 draft/context pairs independently, measure support discrimination,
fix any QA-hypothesis issues, then freeze the method and score full splits.
Use the established training validation split for policy tuning. Keep test data
out of method selection and report verifier computation costs separately until
they are explicitly included in the reward/cost comparison.

The real-example audit and subsequent candidate comparison are documented in
[docs/grounding_audit.md](../docs/grounding_audit.md). The synthetic smoke test alone
did not predict real-data performance.
