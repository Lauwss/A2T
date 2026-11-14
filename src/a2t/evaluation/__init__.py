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
from .listening import BlindStimulus, build_blind_set, build_manifest
from .psychoacoustics import PeaqFftModel
from .probe import (
    ProbeThreshold,
    band_power_at,
    find_probe_threshold,
    probe_power_offset_db,
)

__all__ = [
    "BlindStimulus",
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
    "band_power_at",
    "build_blind_set",
    "build_manifest",
    "compare_conditions",
    "find_probe_threshold",
    "narrowband_noise",
    "pink_noise",
    "probe_power_offset_db",
    "white_noise",
]
