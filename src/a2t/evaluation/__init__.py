"""Evaluation contracts and interpretable masking baselines."""

from .conditions import (
    ConditionComparison,
    compare_conditions,
    narrowband_noise,
    pink_noise,
    white_noise,
)
from .contracts import (
    EvaluationInput,
    EventScore,
    FrameScores,
    MaskingReport,
    RelationRecord,
)
from .masking import MaskingEvaluator, MaskingThresholdEvaluator
from .psychoacoustics import PeaqFftModel
from .probe import ProbeThreshold, find_probe_threshold

__all__ = [
    "ConditionComparison",
    "EvaluationInput",
    "EventScore",
    "FrameScores",
    "MaskingEvaluator",
    "MaskingReport",
    "MaskingThresholdEvaluator",
    "PeaqFftModel",
    "ProbeThreshold",
    "RelationRecord",
    "compare_conditions",
    "find_probe_threshold",
    "narrowband_noise",
    "pink_noise",
    "white_noise",
]
