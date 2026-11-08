"""Probe-threshold estimation for validating the masking-threshold model.

Implements the bisection experiment used by the validation suite: given a
masker, find the level of a steady probe tone at which the model's masking
margin crosses zero. Thresholds are reported as digital-domain amplitudes
(dBFS) plus derived offsets against the masker's RMS power, which is the
quantity the reference psychoacoustic anchors are stated in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .psychoacoustics import PeaqFftModel

NEIGHBOR_BANDS = 2


@dataclass(frozen=True)
class ProbeThreshold:
    """Estimated detection threshold of a probe tone under a fixed masker."""

    probe_hz: float
    threshold_amplitude: float
    threshold_dbfs: float
    masker_rms: float
    power_offset_db: float


def _probe_margin(
    model: PeaqFftModel,
    threshold: np.ndarray,
    probe_hz: float,
    amplitude: float,
    n_samples: int,
) -> float:
    """Worst-case margin of the probe across its neighbouring bands and frames.

    For a steady tone the per-frame margins are nearly constant, so the frame
    median is used inside each band and the maximum across the probe's band
    neighbourhood drives the bisection. This keeps the objective monotonic in
    the probe level.
    """
    t = np.arange(n_samples) / model.sample_rate
    probe = amplitude * np.sin(2.0 * np.pi * probe_hz * t)
    energy = model.band_energy(model.spectrum(probe))
    margins = 10.0 * np.log10(energy / threshold)
    band_hz = model.center_hz
    center_band = int(np.argmin(np.abs(band_hz - probe_hz)))
    lo = max(0, center_band - NEIGHBOR_BANDS)
    hi = min(model.n_bands, center_band + NEIGHBOR_BANDS + 1)
    return float(np.max(np.median(margins[lo:hi, :], axis=1)))


def find_probe_threshold(
    model: PeaqFftModel,
    masker: np.ndarray,
    probe_hz: float,
    lo_dbfs: float = -120.0,
    hi_dbfs: float = 0.0,
    iterations: int = 48,
) -> ProbeThreshold:
    """Bisect the probe amplitude at which the masking margin crosses zero."""
    masker = np.asarray(masker, dtype=float)
    if masker.size < model.n_fft:
        raise ValueError("masker needs at least one full analysis frame")
    threshold = model.masking_threshold(masker)
    masker_rms = float(np.sqrt(np.mean(masker * masker)))

    lo, hi = lo_dbfs, hi_dbfs
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        margin = _probe_margin(model, threshold, probe_hz, 10.0 ** (mid / 20.0), masker.size)
        if margin > 0.0:
            hi = mid
        else:
            lo = mid
    threshold_dbfs = 0.5 * (lo + hi)
    threshold_amplitude = 10.0 ** (threshold_dbfs / 20.0)
    power_offset_db = 10.0 * np.log10(
        (threshold_amplitude**2 / 2.0) / max(masker_rms**2, 1e-30)
    )
    return ProbeThreshold(
        probe_hz=probe_hz,
        threshold_amplitude=float(threshold_amplitude),
        threshold_dbfs=float(threshold_dbfs),
        masker_rms=masker_rms,
        power_offset_db=float(power_offset_db),
    )
