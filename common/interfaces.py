"""Structural interfaces for adapters supplied by the four project roles."""

from typing import Protocol, Sequence
from common.schemas import (
    Action, GenerationResult, Observation, Passage, UncertaintyResult,
    VerificationResult,
)


class Generator(Protocol):
    def generate(self, question: str, passages: Sequence[Passage]) -> GenerationResult: ...


class UncertaintyEstimator(Protocol):
    def estimate(self, generation: GenerationResult) -> UncertaintyResult: ...


class Verifier(Protocol):
    def check_sufficiency(self, question: str, passages: Sequence[Passage]) -> VerificationResult: ...

    def verify_answer(self, question: str, answer: str, passages: Sequence[Passage]) -> VerificationResult: ...


class Policy(Protocol):
    def select_action(self, observation: Observation) -> Action: ...
