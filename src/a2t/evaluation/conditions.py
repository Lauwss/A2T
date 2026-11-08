"""Condition controls for masking evaluation and synthetic masker generators.

The comparison harness produces the four controls required by the project
protocol: the correct pairing, a mismatched pairing, the unconditional
(no-masking) condition, and a traditional pink-noise baseline level-matched to
the candidate masker. Synthetic generators are deterministic under a seed so
runs stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import EvaluationInput, MaskingReport
from .masking import MaskingThresholdEvaluator

CONDITION_NAMES = ("correct", "mismatched", "unconditional", "traditional_pink")


def rms_level(signal: np.ndarray) -> float:
    """Root-mean-square level of a one-dimensional signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    return float(np.sqrt(np.mean(signal * signal)))


def shaped_noise(
    n_samples: int,
    magnitude_shape: np.ndarray,
    rms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian noise shaped by a per-bin magnitude response, normalized to ``rms``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if rms <= 0:
        raise ValueError("rms must be positive")
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    shaped = spectrum * magnitude_shape
    signal = np.fft.irfft(shaped, n=n_samples)
    current = rms_level(signal)
    if current <= 0:
        raise ValueError("shaping produced a silent signal")
    return signal * (rms / current)


def white_noise(n_samples: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    """White noise at the requested RMS level."""
    return shaped_noise(n_samples, np.ones(n_samples // 2 + 1), rms, rng)


def pink_noise(
    n_samples: int,
    rms: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pink (1/f power) noise at the requested RMS level."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    shape = np.ones_like(freqs)
    nonzero = freqs > 0.0
    shape[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    return shaped_noise(n_samples, shape, rms, rng)


def narrowband_noise(
    n_samples: int,
    rms: float,
    sample_rate: int,
    center_hz: float,
    width_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Brickwall band-passed noise at the requested RMS level."""
    if width_hz <= 0 or center_hz <= 0:
        raise ValueError("center_hz and width_hz must be positive")
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    shape = ((freqs >= center_hz - width_hz / 2.0) & (freqs <= center_hz + width_hz / 2.0)).astype(
        float
    )
    if not np.any(shape):
        raise ValueError("band is empty for the given sample rate and width")
    return shaped_noise(n_samples, shape, rms, rng)


@dataclass(frozen=True)
class ConditionComparison:
    """Masking reports for the four protocol conditions of one pair."""

    sample_rate: int
    target_rms: float
    masker_rms: float
    reports: dict[str, MaskingReport]

    def summary(self) -> dict[str, dict[str, float]]:
        """Flat per-condition summary suitable for JSON reports."""
        out: dict[str, dict[str, float]] = {}
        for name, report in self.reports.items():
            out[name] = {
                "masked_frame_ratio": report.masked_frame_ratio,
                "mean_active_margin_db": report.mean_active_margin_db,
                "nmr_total_db": report.nmr_total_db,
                "event_count": float(len(report.events)),
                "min_event_margin_db": (
                    min(event.min_margin_db for event in report.events)
                    if report.events
                    else float("nan")
                ),
            }
        return out


def compare_conditions(
    target: np.ndarray,
    masker: np.ndarray,
    sample_rate: int,
    seed: int = 0,
    playback_db_spl: float = 92.0,
) -> ConditionComparison:
    """Score the correct, mismatched, unconditional, and traditional conditions.

    The mismatched condition time-rolls the masker by half its length (same
    marginal spectrum, different alignment with the target). The traditional
    baseline is pink noise level-matched to the candidate masker's RMS.
    """
    target = np.asarray(target, dtype=float)
    masker = np.asarray(masker, dtype=float)
    if target.size != masker.size:
        raise ValueError("target and masker must have equal lengths")

    evaluator = MaskingThresholdEvaluator(playback_db_spl=playback_db_spl)
    rng = np.random.default_rng(seed)
    masker_rms = rms_level(masker)
    traditional = pink_noise(masker.size, masker_rms, sample_rate, rng)

    conditions = {
        "correct": masker,
        "mismatched": np.roll(masker, masker.size // 2),
        "unconditional": np.zeros_like(masker),
        "traditional_pink": traditional,
    }
    reports = {
        name: evaluator.score(
            EvaluationInput(target, masker_signal, target + masker_signal, sample_rate)
        )
        for name, masker_signal in conditions.items()
    }
    return ConditionComparison(
        sample_rate=sample_rate,
        target_rms=rms_level(target),
        masker_rms=masker_rms,
        reports=reports,
    )
