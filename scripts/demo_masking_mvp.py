"""End-to-end demo of the masking-threshold evaluation MVP.

Builds a synthetic snore-like target and a rain-like masker, runs the four
protocol conditions (correct / mismatched / unconditional / traditional pink
baseline), prints a comparison table and writes a JSON report plus the raw
margin matrices to ``artifacts/masking_mvp/``.

Everything is digital-domain and deterministic (fixed seed). The playback
level assumed by the model (92 dB SPL per full-scale sine) is a modelling
convention, not a statement about real playback loudness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a2t.evaluation import (  # noqa: E402
    ConditionComparison,
    compare_conditions,
    pink_noise,
)

SAMPLE_RATE = 48000
DURATION_S = 6.0
SEED = 20261025
SCORE_VERSION = "peaq-fft-masking-mvp-v1"
PROCESSING_VERSION = "synthetic-demo-v1"
PLAYBACK_DB_SPL = 92.0


def snore_like_target(n_samples: int, sample_rate: int, rms: float) -> np.ndarray:
    """Harmonic buzz at 110 Hz with a slow on/off envelope, plus tonal bursts."""
    t = np.arange(n_samples) / sample_rate
    envelope = ((0.4 * t) % 1.0 < 0.4).astype(float)
    envelope = np.convolve(envelope, np.hanning(8001) / 4000.0, mode="same")
    buzz = np.zeros(n_samples)
    for harmonic in range(1, 9):
        buzz += (1.0 / harmonic) * np.sin(2 * np.pi * 110.0 * harmonic * t)
    buzz *= 0.02 * envelope
    bursts = (((t - 1.5) % 2.5) >= 0.0) & (((t - 1.5) % 2.5) < 0.3)
    tone = 0.006 * np.sin(2 * np.pi * 1000.0 * t) * bursts
    signal = buzz + tone
    return signal * (rms / np.sqrt(np.mean(signal * signal)))


def rain_like_masker(n_samples: int, sample_rate: int, rms: float) -> np.ndarray:
    """Pink noise under a slow two-component amplitude envelope."""
    rng = np.random.default_rng(SEED)
    t = np.arange(n_samples) / sample_rate
    envelope = 0.8 + 0.15 * np.sin(2 * np.pi * 0.17 * t) + 0.05 * np.sin(2 * np.pi * 0.43 * t)
    noise = pink_noise(n_samples, rms, sample_rate, rng)
    scaled = noise * envelope
    return scaled * (rms / np.sqrt(np.mean(scaled * scaled)))


def print_condition_table(comparison: ConditionComparison) -> None:
    headers = (
        "condition",
        "masked frames",
        "mean margin dB",
        "NMR total dB",
        "events",
        "worst event dB",
    )
    rows = []
    summary = comparison.summary()
    for name in ("correct", "mismatched", "unconditional", "traditional_pink"):
        values = summary[name]
        worst = values["min_event_margin_db"]
        rows.append(
            (
                name,
                f"{values['masked_frame_ratio'] * 100:5.1f}%",
                f"{values['mean_active_margin_db']:7.2f}",
                f"{values['nmr_total_db']:7.2f}",
                f"{int(values['event_count']):3d}",
                "n/a" if not np.isfinite(worst) else f"{worst:7.2f}",
            )
        )
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def main() -> None:
    n_samples = int(SAMPLE_RATE * DURATION_S)
    target = snore_like_target(n_samples, SAMPLE_RATE, rms=0.08)
    masker = rain_like_masker(n_samples, SAMPLE_RATE, rms=0.12)
    comparison = compare_conditions(
        target, masker, SAMPLE_RATE, seed=SEED, playback_db_spl=PLAYBACK_DB_SPL
    )

    print("A2T masking-threshold MVP demo")
    print(
        f"target rms {comparison.target_rms:.4f} ({20 * np.log10(comparison.target_rms):.1f} dBFS), "
        f"masker rms {comparison.masker_rms:.4f} ({20 * np.log10(comparison.masker_rms):.1f} dBFS), "
        f"relative level {20 * np.log10(comparison.masker_rms / comparison.target_rms):+.1f} dB"
    )
    print(f"\ncondition comparison ({SCORE_VERSION}):")
    print_condition_table(comparison)

    correct = comparison.reports["correct"]
    print("\ncorrect-condition events:")
    for event in correct.events:
        print(
            f"  frames {event.start_frame:4d}-{event.end_frame:4d} "
            f"({event.start_sample / SAMPLE_RATE:5.2f}-{event.end_sample / SAMPLE_RATE:5.2f} s)  "
            f"masked {event.masked_frame_count}/{event.frame_count}  "
            f"min margin {event.min_margin_db:7.2f} dB"
        )

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "masking_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "score_version": SCORE_VERSION,
        "processing_version": PROCESSING_VERSION,
        "seed": SEED,
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "playback_db_spl_assumption": PLAYBACK_DB_SPL,
        "numpy_version": np.__version__,
        "target_rms": comparison.target_rms,
        "masker_rms": comparison.masker_rms,
        "relative_level_db": float(20 * np.log10(comparison.masker_rms / comparison.target_rms)),
        "conditions": comparison.summary(),
        "correct_events": [
            {
                "start_sample": event.start_sample,
                "end_sample": event.end_sample,
                "frames": event.frame_count,
                "masked_frames": event.masked_frame_count,
                "masked_ratio": event.masked_ratio,
                "min_margin_db": event.min_margin_db,
            }
            for event in correct.events
        ],
        "disclaimer": (
            "Proxy scores in the digital domain; not validated against human "
            "listening and not a claim of masking effectiveness."
        ),
    }
    json_path = out_dir / "demo_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    npz_path = out_dir / "demo_margins.npz"
    np.savez_compressed(
        npz_path,
        band_center_hz=np.asarray(correct.band_center_hz),
        **{
            f"{name}_margins_target_db": report_data.margins_target_db
            for name, report_data in comparison.reports.items()
        },
        **{
            f"{name}_margins_noise_db": report_data.margins_noise_db
            for name, report_data in comparison.reports.items()
        },
    )
    print(f"\nwritten: {json_path}")
    print(f"written: {npz_path}")


if __name__ == "__main__":
    main()
