"""Stage 2 synthetic-only relevance classifier. Live inference is locked."""

from .evaluate import EvaluationError, evaluate_gate, to_classifier_inputs
from .gate_prepare import GATE_REAL_RUN_LOCKED, GateRunLockedError, describe_provider_payload
from .pipeline import (
    CheckpointStore,
    DuplicateIdError,
    HumanCorrectionStore,
    assert_no_raw_text,
    classify_records,
    public_row,
)
from .planner import plan_batches, simulate_full_frame_budget
from .preprocess import ProtectedOutcomeSuppliedError, preprocess_record
from .providers import LiveInferenceForbidden, LiveOpenAIProvider, MockProvider, MockTurn
from .spec import STAGE2_LIVE_LOCKED

__all__ = [
    "CheckpointStore",
    "DuplicateIdError",
    "EvaluationError",
    "GATE_REAL_RUN_LOCKED",
    "GateRunLockedError",
    "HumanCorrectionStore",
    "LiveInferenceForbidden",
    "LiveOpenAIProvider",
    "MockProvider",
    "MockTurn",
    "ProtectedOutcomeSuppliedError",
    "STAGE2_LIVE_LOCKED",
    "assert_no_raw_text",
    "classify_records",
    "describe_provider_payload",
    "evaluate_gate",
    "plan_batches",
    "preprocess_record",
    "public_row",
    "simulate_full_frame_budget",
    "to_classifier_inputs",
]
