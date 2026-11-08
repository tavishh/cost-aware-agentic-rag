"""Version 1 inference contracts. Gold labels deliberately live outside these types."""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Optional, Tuple

SCHEMA_VERSION = 1


class Action(str, Enum):
    ANSWER = "ANSWER"
    RETRIEVE = "RETRIEVE"
    REWRITE = "REWRITE"
    ABSTAIN = "ABSTAIN"


class Verdict(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Passage:
    docid: str
    text: str
    score: Optional[float] = None

    def __post_init__(self):
        if not self.docid:
            raise ValueError("docid must not be empty")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("retrieval score must be finite")


@dataclass(frozen=True)
class Cost:
    retrieval_calls: int = 0
    generation_calls: int = 0
    verifier_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            _nonnegative(value, name)
            if name != "latency_ms" and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer")


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model_version: str
    prompt_version: str
    # Selected answer-token log probabilities, excluding prompt and padding tokens.
    token_logprobs: Optional[Tuple[float, ...]] = None
    cost: Cost = Cost()

    def __post_init__(self):
        if not self.model_version or not self.prompt_version:
            raise ValueError("generation model and prompt versions are required")
        if self.token_logprobs is not None:
            if not self.token_logprobs or any(not math.isfinite(x) or x > 0 for x in self.token_logprobs):
                raise ValueError("token_logprobs must be nonempty, finite and <= 0; use None if unavailable")


@dataclass(frozen=True)
class UncertaintyResult:
    method: str
    value: Optional[float]
    unavailable_reason: Optional[str] = None

    def __post_init__(self):
        if not self.method:
            raise ValueError("uncertainty method is required")
        if self.value is None:
            if not self.unavailable_reason:
                raise ValueError("missing uncertainty requires a reason")
        else:
            _nonnegative(self.value, "uncertainty")
            if self.unavailable_reason is not None:
                raise ValueError("available uncertainty cannot have an unavailable reason")


@dataclass(frozen=True)
class VerificationResult:
    # For sufficiency: YES means the context can answer the question.
    # For grounding: YES means all factual content of the draft is supported.
    verdict: Verdict
    evidence_ids: Tuple[str, ...]
    explanation: str
    verifier_version: str
    prompt_version: str
    score: Optional[float] = None
    cost: Cost = Cost()

    def __post_init__(self):
        if not isinstance(self.verdict, Verdict):
            raise ValueError("verdict must be a Verdict")
        if not self.verifier_version or not self.prompt_version or not self.explanation:
            raise ValueError("verifier versions and explanation are required")
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("score must be finite and in [0, 1]")
        if self.verdict is Verdict.UNKNOWN and self.score is not None:
            raise ValueError("unknown verdict must not carry a support score")


@dataclass(frozen=True)
class Observation:
    qid: str
    question: str
    query: str
    passages: Tuple[Passage, ...]
    generation: GenerationResult
    uncertainty: UncertaintyResult
    action_history: Tuple[Action, ...] = ()
    remaining_steps: int = 3
    sufficiency: Optional[VerificationResult] = None
    grounding: Optional[VerificationResult] = None
    cumulative_cost: Cost = Cost()

    def __post_init__(self):
        if not self.qid or not self.question or not self.query:
            raise ValueError("question identity and query are required")
        if isinstance(self.remaining_steps, bool) or not isinstance(self.remaining_steps, int) or self.remaining_steps < 0:
            raise ValueError("remaining_steps must be a nonnegative integer")
        if any(a not in (Action.RETRIEVE, Action.REWRITE) for a in self.action_history):
            raise ValueError("observations may contain only nonterminal action history")
        ids = [p.docid for p in self.passages]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate passage IDs")
        for result in (self.sufficiency, self.grounding):
            if result is not None and not set(result.evidence_ids).issubset(ids):
                raise ValueError("verifier cited evidence outside the observation")

    def to_dict(self):
        """A JSON-compatible, inference-only payload (never an offline cache row)."""
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}
