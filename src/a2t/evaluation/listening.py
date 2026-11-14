"""Blind-listening stimulus construction around estimated thresholds.

The probe experiment returns a detection threshold in dBFS. To check that
number by ear, this module builds matched stimuli: the masker plus a steady
probe tone at several levels around the threshold. Labels are shuffled across
levels so the listener does not know which level they are hearing; the answer
key is returned to the caller and written out separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LISTENING_OFFSETS_DB = (-6.0, -3.0, 0.0, 3.0, 6.0)
LISTENING_LABELS = ("A", "B", "C", "D", "E")
FADE_S = 0.02
PEAK_LIMIT = 0.999


@dataclass(frozen=True)
class BlindStimulus:
    """One masker-plus-probe file in the blind listening set."""

    filename: str
    probe_hz: float
    label: str
    level_dbfs: float
    signal: np.ndarray


def faded_tone(
    freq_hz: float,
    amplitude: float,
    n_samples: int,
    sample_rate: int,
    fade_s: float = FADE_S,
) -> np.ndarray:
    """Steady sine with linear fade-in/out to avoid onset and offset clicks."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq_hz <= 0 or freq_hz >= sample_rate / 2.0:
        raise ValueError("freq_hz must lie between 0 and Nyquist")

    t = np.arange(n_samples) / float(sample_rate)
    tone = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    n_fade = min(int(round(fade_s * sample_rate)), n_samples // 2)
    if n_fade > 1:
        ramp = np.linspace(0.0, 1.0, n_fade, endpoint=False)
        tone[:n_fade] *= ramp
        tone[-n_fade:] *= ramp[::-1]
    return tone


def build_blind_set(
    masker: np.ndarray,
    sample_rate: int,
    thresholds_dbfs: dict[float, float],
    seed: int = 0,
    offsets_db: tuple[float, ...] = LISTENING_OFFSETS_DB,
) -> tuple[tuple[BlindStimulus, ...], float]:
    """Build shuffled masker-plus-probe stimuli around each threshold.

    Every probe frequency gets one stimulus per offset, with offsets assigned
    to labels at random under ``seed``. A single global gain is applied to all
    stimuli when the masker peak plus the loudest probe would exceed
    :data:`PEAK_LIMIT`, so the relative levels stay intact.

    Returns the stimuli and the applied gain (1.0 when no scaling was needed).
    Levels in the returned stimuli are post-gain.
    """
    masker = np.asarray(masker, dtype=float)
    if masker.ndim != 1 or masker.size == 0:
        raise ValueError("masker must be a non-empty one-dimensional signal")
    if not np.all(np.isfinite(masker)):
        raise ValueError("masker must be finite")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not thresholds_dbfs:
        raise ValueError("thresholds_dbfs must not be empty")
    if len(offsets_db) > len(LISTENING_LABELS):
        raise ValueError("more offsets than available labels")

    rng = np.random.default_rng(seed)
    loudest_probe = max(
        10.0 ** ((threshold + max(offsets_db)) / 20.0)
        for threshold in thresholds_dbfs.values()
    )
    peak = float(np.max(np.abs(masker))) + loudest_probe
    gain = PEAK_LIMIT / peak if peak > PEAK_LIMIT else 1.0
    gain_db = 20.0 * np.log10(gain)

    stimuli: list[BlindStimulus] = []
    for probe_hz in sorted(thresholds_dbfs):
        levels = [thresholds_dbfs[probe_hz] + offset for offset in offsets_db]
        rng.shuffle(levels)
        for label, level in zip(LISTENING_LABELS, levels):
            tone = faded_tone(probe_hz, 10.0 ** (level / 20.0), masker.size, sample_rate)
            stimuli.append(
                BlindStimulus(
                    filename=f"f{probe_hz:g}_{label}.wav",
                    probe_hz=float(probe_hz),
                    label=label,
                    level_dbfs=float(level + gain_db),
                    signal=(masker + tone) * gain,
                )
            )
    return tuple(stimuli), float(gain)


def build_manifest(
    masker_only: str,
    stimuli: Sequence[BlindStimulus],
    sample_rate: int,
    masker: str = "",
) -> dict:
    """Public test manifest: which files exist, never their levels.

    The levels stay in ``answers.json`` so the browser page can run the test
    blind and fetch the key only when scoring. Labels are visible because the
    shuffle already hides which label carries which level.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    grouped: dict[float, dict] = {}
    for stimulus in stimuli:
        entry = grouped.setdefault(
            stimulus.probe_hz, {"hz": stimulus.probe_hz, "stimuli": []}
        )
        entry["stimuli"].append({"file": stimulus.filename, "label": stimulus.label})
    return {
        "version": 1,
        "masker": masker,
        "sample_rate": int(sample_rate),
        "masker_only": masker_only,
        "responses": ("yes", "unsure", "no"),
        "frequencies": [grouped[hz] for hz in sorted(grouped)],
    }
