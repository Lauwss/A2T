"""Data contracts for evaluation outputs.

The masking-threshold contracts carry NumPy arrays for the frame-by-band
matrices; every other contract stays dependency-free.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationInput:
    """Three aligned digital-domain signals used only for evaluation."""

    target: Sequence[float]
    masker: Sequence[float]
    mixture: Sequence[float]
    sample_rate: int


@dataclass(frozen=True)
class FrameScores:
    """Frame-level scores retained for later frequency-band aggregation."""

    threshold_exposure_db: tuple[float, ...]
    representation_distance_db: tuple[float, ...]
    frame_hop_samples: int


@dataclass(frozen=True)
class RelationRecord:
    """Metadata linking a target, candidate masker, and evaluation run."""

    relation_id: str
    target_id: str
    candidate_id: str
    relative_level_db: float
    processing_version: str
    score_version: str


@dataclass(frozen=True)
class EventScore:
    """Event-level outcome for one contiguous span of target-active frames."""

    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    frame_count: int
    masked_frame_count: int
    masked_ratio: float
    min_margin_db: float


@dataclass(frozen=True)
class MaskingReport:
    """Full masking-threshold evaluation of one target/masker pair.

    All levels are digital-domain dB relative to full scale under the model's
    playback-level assumption; they are proxy quantities, not measured sound
    pressure. ``margins_*`` are frame-by-band matrices shaped
    ``(n_bands, n_frames)``; positive margin means the target (or the
    masker-vs-mixture difference) exceeds the masking threshold there.
    """

    sample_rate: int
    frame_hop_samples: int
    n_frames: int
    playback_db_spl: float
    score_version: str
    band_center_hz: tuple[float, ...]
    margins_target_db: np.ndarray
    margins_noise_db: np.ndarray
    frame_active: np.ndarray
    frame_audible: np.ndarray
    frame_margin_target_db: np.ndarray
    band_active_frame_count: np.ndarray
    band_masked_ratio: np.ndarray
    events: tuple[EventScore, ...]
    masked_frame_ratio: float
    mean_active_margin_db: float
    nmr_local_db: np.ndarray
    nmr_total_db: float
    relation: RelationRecord | None = None
