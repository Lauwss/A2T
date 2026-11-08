"""Validation experiments for the masking-threshold model MVP.

Runs six deterministic probe experiments against published psychoacoustic
anchors taken from the project reference document (ITU-R BS.1387-2 Chinese
translation) and prints a summary table. Results are written to
``artifacts/masking_mvp/validation_results.json``.

Experiments:
  E1  threshold in quiet vs the loudness-model anchor (eq 60) at 0.5-8 kHz;
  E2  pink-noise-masking-tone offset vs the ~-5 dB anchor (source section 3.5);
  E3  narrowband-noise-masking-tone offset (documented broadband-scope check);
  E4  frequency-spreading slopes vs the stated 27 dB/Bark down / level-dependent
      up slopes (eq 15-16);
  E5  forward masking decay after masker offset (eq 21-24);
  E6  threshold growth versus masker level (expect ~1 dB per dB).

All quantities are digital-domain; the SPL-equivalent column uses the model's
playback-level assumption (92 dB SPL per full-scale sine) and is NOT a claim
about real playback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a2t.evaluation.conditions import narrowband_noise, pink_noise  # noqa: E402
from a2t.evaluation.psychoacoustics import (  # noqa: E402
    PeaqFftModel,
    ear_weight_db,
    hz_to_bark,
)
from a2t.evaluation.probe import find_probe_threshold  # noqa: E402

SAMPLE_RATE = 48000
DURATION_S = 1.5
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
MASKER_RMS = 0.1
PLAYBACK_DB = 92.0
SEED = 20261025


def spl_equivalent(model: PeaqFftModel, freq_hz: float, amplitude: float) -> float:
    """Digital amplitude -> playback-assumed SPL of an equal full-scale sine."""
    weight = float(ear_weight_db(freq_hz))
    return PLAYBACK_DB + 20.0 * np.log10(max(amplitude, 1e-12)) + weight


def band_power_at(model: PeaqFftModel, masker: np.ndarray, freq_hz: float) -> float:
    """Mean model band energy at ``freq_hz`` converted to physical mean-square power."""
    energy = model.band_energy(model.spectrum(masker))
    band = int(np.argmin(np.abs(model.center_hz - freq_hz)))
    weight = 10.0 ** (float(ear_weight_db(model.center_hz[band])) / 10.0)
    return float(energy[band, :].mean()) / (model.fac**2 * weight)


def experiment_quiet_threshold(model: PeaqFftModel) -> list[dict[str, float]]:
    silence = np.zeros(N_SAMPLES)
    rows = []
    for freq in (500.0, 1000.0, 2000.0, 4000.0, 8000.0):
        probe = find_probe_threshold(model, silence, freq)
        spl = spl_equivalent(model, freq, probe.threshold_amplitude)
        anchor = 0.364 * (freq / 1000.0) ** -0.8  # eq (60) loudness-model threshold
        rows.append(
            {
                "freq_hz": freq,
                "threshold_dbfs": probe.threshold_dbfs,
                "spl_equivalent_db": spl,
                "eq60_anchor_db": anchor,
                "delta_db": spl - anchor,
            }
        )
    return rows


def experiment_pink_masking_offset(model: PeaqFftModel) -> list[dict[str, float]]:
    rng = np.random.default_rng(SEED)
    masker = pink_noise(N_SAMPLES, MASKER_RMS, SAMPLE_RATE, rng)
    rows = []
    for freq in (500.0, 1000.0, 2000.0, 4000.0):
        probe = find_probe_threshold(model, masker, freq)
        band_power = band_power_at(model, masker, freq)
        offset = 10.0 * np.log10((probe.threshold_amplitude**2 / 2.0) / band_power)
        rows.append(
            {
                "freq_hz": freq,
                "threshold_dbfs": probe.threshold_dbfs,
                "in_band_power_db": 10.0 * np.log10(band_power),
                "power_offset_db": offset,
                "anchor_db": -5.0,
                "delta_db": offset + 5.0,
            }
        )
    return rows


def experiment_narrowband_masking_offset(model: PeaqFftModel) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    masker = narrowband_noise(
        N_SAMPLES, MASKER_RMS, SAMPLE_RATE, center_hz=1000.0, width_hz=100.0, rng=rng
    )
    probe = find_probe_threshold(model, masker, 1000.0)
    band_power = band_power_at(model, masker, 1000.0)
    offset = 10.0 * np.log10((probe.threshold_amplitude**2 / 2.0) / band_power)
    return {
        "freq_hz": 1000.0,
        "masker_width_hz": 100.0,
        "threshold_dbfs": probe.threshold_dbfs,
        "in_band_power_db": 10.0 * np.log10(band_power),
        "power_offset_db": offset,
        "anchor_db": -5.0,
        "delta_db": offset + 5.0,
    }


def experiment_spreading_slopes(model: PeaqFftModel) -> dict[str, float]:
    """Fit the decay of the masking pattern itself around a localized masker.

    Measured on the model's masking threshold (no probe), so the fit reflects
    the spreading kernel of eq (15)-(16) without probe-side confounds such as
    the absolute-threshold floor.
    """
    rng = np.random.default_rng(SEED + 2)
    center_hz = 500.0
    masker = narrowband_noise(
        N_SAMPLES, 0.316, SAMPLE_RATE, center_hz=center_hz, width_hz=60.0, rng=rng
    )
    threshold = model.masking_threshold(masker)
    center_band = int(np.argmin(np.abs(model.center_hz - center_hz)))
    band_z = np.array([hz_to_bark(f) for f in model.center_hz])
    z0 = band_z[center_band]
    level_db = 10.0 * np.log10(threshold[:, threshold.shape[1] // 2])
    points = []
    for offset_bark in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0):
        band = int(np.argmin(np.abs(band_z - (z0 + offset_bark))))
        points.append(
            {
                "offset_bark": offset_bark,
                "freq_hz": float(model.center_hz[band]),
                "threshold_db": float(level_db[band]),
            }
        )
    up = [p for p in points if p["offset_bark"] >= 1.0]
    down = [p for p in points if -3.0 <= p["offset_bark"] <= -1.0]
    up_slope = float(
        np.polyfit([p["offset_bark"] for p in up], [p["threshold_db"] for p in up], 1)[0]
    )
    down_slope = float(
        np.polyfit([p["offset_bark"] for p in down], [p["threshold_db"] for p in down], 1)[0]
    )
    return {
        "points": points,
        "upward_slope_db_per_bark": up_slope,
        "downward_slope_db_per_bark": down_slope,
        "masker_band_level_db": float(level_db[center_band]),
    }


def experiment_forward_masking(model: PeaqFftModel) -> list[dict[str, float]]:
    rng = np.random.default_rng(SEED + 3)
    mask_len = SAMPLE_RATE  # 1.0 s of masker, then silence
    masker = narrowband_noise(
        mask_len, MASKER_RMS, SAMPLE_RATE, center_hz=1000.0, width_hz=160.0, rng=rng
    )
    signal = np.concatenate([masker, np.zeros(N_SAMPLES - mask_len)])
    threshold = model.masking_threshold(signal)
    probe_amplitude = 10.0 ** (-60.0 / 20.0)
    burst_len = int(0.1 * SAMPLE_RATE)
    t_burst = np.arange(burst_len) / SAMPLE_RATE
    burst = probe_amplitude * np.sin(2.0 * np.pi * 1000.0 * t_burst)
    band = int(np.argmin(np.abs(model.center_hz - 1000.0)))
    rows = []
    for delay_ms in (0.0, 50.0, 150.0, 300.0):
        probe_signal = np.zeros(N_SAMPLES)
        start = mask_len + int(delay_ms * SAMPLE_RATE / 1000.0)
        probe_signal[start : start + burst_len] = burst
        energy = model.band_energy(model.spectrum(probe_signal))
        margin = 10.0 * np.log10(energy / threshold)
        frames = slice(start // model.hop, min((start + burst_len) // model.hop + 1, margin.shape[1]))
        rows.append(
            {
                "delay_ms": delay_ms,
                "margin_db": float(np.max(margin[band, frames])),
            }
        )
    return rows


def experiment_level_linearity(model: PeaqFftModel) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 4)
    levels_db = (-30.0, -20.0, -10.0)
    thresholds = []
    for level_db in levels_db:
        masker = pink_noise(N_SAMPLES, 10.0 ** (level_db / 20.0), SAMPLE_RATE, rng)
        probe = find_probe_threshold(model, masker, 1000.0)
        thresholds.append(probe.threshold_dbfs)
    slope = float(np.polyfit(levels_db, thresholds, 1)[0])
    return {
        "masker_levels_dbfs_rms": list(levels_db),
        "threshold_dbfs": thresholds,
        "growth_db_per_db": slope,
    }


def main() -> None:
    model = PeaqFftModel(sample_rate=SAMPLE_RATE, playback_db_spl=PLAYBACK_DB)
    results = {
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "masker_rms": MASKER_RMS,
        "seed": SEED,
        "numpy_version": np.__version__,
        "quiet_threshold": experiment_quiet_threshold(model),
        "pink_masking_offset": experiment_pink_masking_offset(model),
        "narrowband_masking_offset": experiment_narrowband_masking_offset(model),
        "spreading_slopes": experiment_spreading_slopes(model),
        "forward_masking": experiment_forward_masking(model),
        "level_linearity": experiment_level_linearity(model),
    }

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "masking_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"results written to {out_path}")
    print("\nE1 threshold in quiet (SPL-equivalent vs eq60 anchor):")
    for row in results["quiet_threshold"]:
        print(
            f"  {row['freq_hz']:7.0f} Hz  {row['spl_equivalent_db']:8.2f} dB"
            f"  anchor {row['eq60_anchor_db']:6.2f} dB  delta {row['delta_db']:+7.2f} dB"
        )
    print("\nE2 pink-noise-masking-tone offset (anchor -5 dB):")
    for row in results["pink_masking_offset"]:
        print(
            f"  {row['freq_hz']:7.0f} Hz  offset {row['power_offset_db']:8.2f} dB"
            f"  delta {row['delta_db']:+7.2f} dB"
        )
    nb = results["narrowband_masking_offset"]
    print(
        f"\nE3 narrowband (100 Hz) masking offset at 1 kHz: "
        f"{nb['power_offset_db']:.2f} dB (anchor -5 dB, delta {nb['delta_db']:+.2f} dB)"
    )
    sp = results["spreading_slopes"]
    print(
        f"\nE4 spreading slopes: upward {sp['upward_slope_db_per_bark']:.2f} dB/Bark, "
        f"downward {sp['downward_slope_db_per_bark']:.2f} dB/Bark"
    )
    print("\nE5 forward masking (probe margin after masker offset):")
    for row in results["forward_masking"]:
        print(f"  +{row['delay_ms']:6.1f} ms  margin {row['margin_db']:8.2f} dB")
    lin = results["level_linearity"]
    print(f"\nE6 threshold growth vs masker level: {lin['growth_db_per_db']:.3f} dB/dB")


if __name__ == "__main__":
    main()
