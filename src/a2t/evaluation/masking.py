"""Masking evaluation: legacy energy-ratio baseline and the PEAQ-based model.

``MaskingEvaluator`` is the original frame-level relative-energy proxy kept as
a transparent baseline. ``MaskingThresholdEvaluator`` implements the
project's masking-threshold model on top of the PEAQ FFT ear model
(:mod:`a2t.evaluation.psychoacoustics`): it derives the masking threshold from
the masker's excitation pattern and compares the target (and the
masker-versus-mixture difference pattern) against it, keeping frame-level,
band-level and event-level results.

Both are digital-domain proxies; neither is a human-effectiveness claim.
"""

from math import log10
from typing import Sequence

import numpy as np

from .contracts import (
    EvaluationInput,
    EventScore,
    FrameScores,
    MaskingReport,
    RelationRecord,
)
from .psychoacoustics import ENERGY_FLOOR, PeaqFftModel

SCORE_VERSION = "peaq-fft-masking-mvp-v1"


class MaskingEvaluator:
    """Compute a simple frame-level relative-energy proxy."""

    def __init__(self, *, frame_size: int = 2048, hop_size: int = 1024) -> None:
        if frame_size <= 0 or hop_size <= 0:
            raise ValueError("frame_size and hop_size must be positive")
        self.frame_size = frame_size
        self.hop_size = hop_size

    def score(self, inputs: EvaluationInput) -> FrameScores:
        """Return uncalibrated target-to-masker energy ratios per frame."""
        self._validate_inputs(inputs)
        threshold_scores: list[float] = []
        representation_scores: list[float] = []
        for start in range(0, len(inputs.target) - self.frame_size + 1, self.hop_size):
            target_frame = inputs.target[start : start + self.frame_size]
            masker_frame = inputs.masker[start : start + self.frame_size]
            mixture_frame = inputs.mixture[start : start + self.frame_size]
            target_energy = self._mean_square(target_frame)
            masker_energy = self._mean_square(masker_frame)
            mixture_energy = self._mean_square(mixture_frame)
            threshold_scores.append(10.0 * log10((target_energy + 1e-12) / (masker_energy + 1e-12)))
            representation_scores.append(
                abs(10.0 * log10((mixture_energy + 1e-12) / (masker_energy + 1e-12)))
            )
        return FrameScores(
            threshold_exposure_db=tuple(threshold_scores),
            representation_distance_db=tuple(representation_scores),
            frame_hop_samples=self.hop_size,
        )

    def _validate_inputs(self, inputs: EvaluationInput) -> None:
        if inputs.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        lengths = {len(inputs.target), len(inputs.masker), len(inputs.mixture)}
        if len(lengths) != 1:
            raise ValueError("target, masker, and mixture must have equal lengths")
        if len(inputs.target) < self.frame_size:
            raise ValueError("signals must contain at least one complete frame")

    @staticmethod
    def _mean_square(samples: Sequence[float]) -> float:
        return sum(sample * sample for sample in samples) / len(samples)


class MaskingThresholdEvaluator:
    """Evaluate how well a masker hides a target, per band, frame, and event.

    The masking threshold ``M`` is computed from the masker alone (PEAQ FFT
    ear model). Two margins are reported:

    - ``margins_target_db``: target band energy versus ``M`` (primary);
    - ``margins_noise_db``: the masker-versus-mixture difference pattern
      versus ``M`` (PEAQ's noise pattern, eq 62), aggregated into NMR
      (eq 70-71).

    A band-frame counts as target-active when the target's band energy exceeds
    the model's internal-noise floor; a frame is audible when any active band
    exceeds the threshold. The evaluator is a proxy and must not be read as a
    human-detection claim.
    """

    def __init__(self, playback_db_spl: float = 92.0) -> None:
        self.playback_db_spl = float(playback_db_spl)
        self._models: dict[int, PeaqFftModel] = {}

    def model_for(self, sample_rate: int) -> PeaqFftModel:
        model = self._models.get(sample_rate)
        if model is None:
            model = PeaqFftModel(sample_rate=sample_rate, playback_db_spl=self.playback_db_spl)
            self._models[sample_rate] = model
        return model

    def score(
        self,
        inputs: EvaluationInput,
        relation: RelationRecord | None = None,
    ) -> MaskingReport:
        model = self.model_for(inputs.sample_rate)
        target = np.asarray(inputs.target, dtype=float)
        masker = np.asarray(inputs.masker, dtype=float)
        mixture = np.asarray(inputs.mixture, dtype=float)
        self._validate_inputs(inputs, target, masker, mixture)

        threshold = model.masking_threshold(masker)
        target_energy = model.band_energy(model.spectrum(target))
        masker_weighted = np.abs(model.spectrum(masker))
        mixture_weighted = np.abs(model.spectrum(mixture))
        noise_energy = model.band_energy(np.abs(masker_weighted - mixture_weighted))

        margins_target_db = 10.0 * np.log10(target_energy / threshold)
        margins_noise_db = 10.0 * np.log10(noise_energy / threshold)

        active = target_energy > model.p_threshold[:, None]
        audible = active & (margins_target_db > 0.0)
        frame_active = active.any(axis=0)
        frame_audible = audible.any(axis=0)
        frame_margin = np.where(
            frame_active,
            np.max(np.where(active, margins_target_db, -np.inf), axis=0),
            np.nan,
        )

        band_active_count = active.sum(axis=1).astype(np.int64)
        band_active_total = band_active_count.astype(float)
        band_masked_ratio = np.divide(
            (active & ~audible).sum(axis=1),
            band_active_total,
            out=np.zeros_like(band_active_total),
            where=band_active_total > 0,
        )

        events = self._segment_events(
            frame_active=frame_active,
            frame_audible=frame_audible,
            margins_target_db=margins_target_db,
            active=active,
            hop=model.hop,
            signal_length=target.size,
        )

        active_count = int(frame_active.sum())
        masked_frame_ratio = (
            float((frame_active & ~frame_audible).sum()) / active_count
            if active_count
            else 0.0
        )
        mean_active_margin_db = (
            float(margins_target_db[active].mean()) if active.any() else 0.0
        )

        nmr_ratio = noise_energy / threshold
        nmr_local_db = 10.0 * np.log10(nmr_ratio.mean(axis=0))
        nmr_total_db = float(10.0 * np.log10(nmr_ratio.mean()))

        return MaskingReport(
            sample_rate=inputs.sample_rate,
            frame_hop_samples=model.hop,
            n_frames=int(target_energy.shape[1]),
            playback_db_spl=self.playback_db_spl,
            score_version=SCORE_VERSION,
            band_center_hz=tuple(float(f) for f in model.center_hz),
            margins_target_db=margins_target_db,
            margins_noise_db=margins_noise_db,
            frame_active=frame_active,
            frame_audible=frame_audible,
            frame_margin_target_db=frame_margin,
            band_active_frame_count=band_active_count,
            band_masked_ratio=band_masked_ratio,
            events=events,
            masked_frame_ratio=masked_frame_ratio,
            mean_active_margin_db=mean_active_margin_db,
            nmr_local_db=nmr_local_db,
            nmr_total_db=nmr_total_db,
            relation=relation,
        )

    def _segment_events(
        self,
        frame_active: np.ndarray,
        frame_audible: np.ndarray,
        margins_target_db: np.ndarray,
        active: np.ndarray,
        hop: int,
        signal_length: int,
    ) -> tuple[EventScore, ...]:
        events: list[EventScore] = []
        n_frames = frame_active.size
        index = 0
        while index < n_frames:
            if not frame_active[index]:
                index += 1
                continue
            end = index
            while end < n_frames and frame_active[end]:
                end += 1
            window = slice(index, end)
            event_margins = margins_target_db[:, window][active[:, window]]
            events.append(
                EventScore(
                    start_frame=index,
                    end_frame=end,
                    start_sample=index * hop,
                    end_sample=min(end * hop + 1, signal_length),
                    frame_count=end - index,
                    masked_frame_count=int((~frame_audible[window]).sum()),
                    masked_ratio=float((~frame_audible[window]).mean()),
                    min_margin_db=float(event_margins.min()),
                )
            )
            index = end
        return tuple(events)

    @staticmethod
    def _validate_inputs(
        inputs: EvaluationInput,
        target: np.ndarray,
        masker: np.ndarray,
        mixture: np.ndarray,
    ) -> None:
        if inputs.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not (target.size == masker.size == mixture.size):
            raise ValueError("target, masker, and mixture must have equal lengths")
        for name, signal in (("target", target), ("masker", masker), ("mixture", mixture)):
            if not np.all(np.isfinite(signal)):
                raise ValueError(f"{name} signal must be finite")
        if target.size < 1:
            raise ValueError("signals must not be empty")
