"""Probe-threshold test harness with optional blind-listening export.

Runs the bisection probe experiment against a masker you choose:

  silence      absolute threshold in quiet;
  white        flat-spectrum noise;
  pink         1/f noise (rain-like);
  narrowband   brick-wall band-passed noise;
  file         any integer-PCM WAV, analysed at its own sample rate.

For every probe frequency the script prints the detection threshold in dBFS
and the threshold power relative to the masker's power in the probe's band --
the quantity the reference anchors are stated in.

``--export DIR`` writes a blind-listening set: the masker alone plus one file
per probe level around the estimated threshold, shuffled over labels A..E,
with the answer key in ``answers.json``. That is the check the numbers cannot
provide: whether the boundary the model predicts is the boundary you hear.

Everything stays in the digital domain. The playback assumption (92 dB SPL per
full-scale sine) is a modelling convention, not a claim about real loudness.

Examples
--------
    .venv\\Scripts\\python.exe scripts/probe_test.py
    .venv\\Scripts\\python.exe scripts/probe_test.py --masker narrowband \\
        --narrow-center 1000 --narrow-width 100 --freqs 1000
    .venv\\Scripts\\python.exe scripts/probe_test.py --masker file \\
        --masker-file Datasets/rain.wav --freqs 1000,2000 --export artifacts/probe_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a2t.data.wav import read_wav_mono, write_wav_24  # noqa: E402
from a2t.evaluation import (  # noqa: E402
    PeaqFftModel,
    band_power_at,
    build_blind_set,
    build_manifest,
    find_probe_threshold,
    narrowband_noise,
    pink_noise,
    probe_power_offset_db,
    white_noise,
)
from a2t.evaluation.conditions import rms_level  # noqa: E402

DEFAULT_FREQS = "500,1000,2000,4000"
DEFAULT_OFFSETS = "-6,-3,0,3,6"
MIN_SAMPLE_RATE = 36000
QUIET_SWEEP_CEILING_DBFS = -0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate probe detection thresholds under a chosen masker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The output is a model prediction in the digital domain, not a\n"
            "measured human threshold. Use --export to check it by ear."
        ),
    )
    parser.add_argument("--masker", default="pink", choices=("silence", "white", "pink", "narrowband", "file"))
    parser.add_argument("--masker-file", type=Path, help="WAV source when --masker file")
    parser.add_argument(
        "--normalise-file",
        action="store_true",
        help="rescale a --masker file to --level instead of keeping its level",
    )
    parser.add_argument("--level", type=float, default=-20.0, help="masker RMS level in dBFS (synthetic maskers)")
    parser.add_argument("--narrow-center", type=float, default=1000.0, help="narrowband masker centre in Hz")
    parser.add_argument("--narrow-width", type=float, default=100.0, help="narrowband masker width in Hz")
    parser.add_argument("--freqs", default=DEFAULT_FREQS, help="comma-separated probe frequencies in Hz")
    parser.add_argument("--duration", type=float, default=1.5, help="analysis duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=48000, help="rate for synthetic maskers and probes")
    parser.add_argument("--playback-db", type=float, default=92.0, help="assumed SPL of a full-scale sine")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--offsets",
        default=DEFAULT_OFFSETS,
        help="comma-separated probe offsets in dB around the threshold for --export",
    )
    parser.add_argument("--export", type=Path, help="directory for the blind listening set")
    parser.add_argument("--json", type=Path, help="write the threshold table to this JSON file")
    args = parser.parse_args(argv)
    if args.masker == "file" and args.masker_file is None:
        parser.error("--masker file requires --masker-file")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def parse_floats(text: str, name: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one number")
    return values


def normalise_rms(signal: np.ndarray, target_rms: float) -> np.ndarray:
    current = rms_level(signal)
    if current <= 0.0:
        raise SystemExit("cannot normalise a silent signal")
    return signal * (target_rms / current)


def centre_crop(signal: np.ndarray, duration_s: float, sample_rate: int) -> np.ndarray:
    n_samples = int(round(duration_s * sample_rate))
    if signal.size <= n_samples:
        return signal
    start = (signal.size - n_samples) // 2
    return signal[start : start + n_samples]


def build_masker(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, int, str]:
    """Return ``(masker, sample_rate, description)`` for the requested source."""
    if args.masker == "file":
        signal, sample_rate = read_wav_mono(args.masker_file)
        signal = centre_crop(signal, args.duration, sample_rate)
        if args.normalise_file:
            signal = normalise_rms(signal, 10.0 ** (args.level / 20.0))
        described = f"file {args.masker_file.name}"
        return signal, sample_rate, described

    sample_rate = args.sample_rate
    n_samples = int(round(args.duration * sample_rate))
    level = 10.0 ** (args.level / 20.0)
    if args.masker == "silence":
        return np.zeros(n_samples), sample_rate, "silence"
    if args.masker == "white":
        return white_noise(n_samples, level, rng), sample_rate, "white noise"
    if args.masker == "pink":
        return pink_noise(n_samples, level, sample_rate, rng), sample_rate, "pink noise"
    if args.masker == "narrowband":
        masker = narrowband_noise(
            n_samples,
            level,
            sample_rate,
            center_hz=args.narrow_center,
            width_hz=args.narrow_width,
            rng=rng,
        )
        return masker, sample_rate, f"narrowband {args.narrow_center:g} Hz +/- {args.narrow_width / 2:g} Hz"
    raise AssertionError(f"unhandled masker {args.masker}")


def print_table(rows: list[dict[str, float]]) -> None:
    print(f"  {'probe Hz':>9}  {'threshold dBFS':>14}  {'masker band dB':>14}  {'offset dB':>9}")
    for row in rows:
        band = "n/a" if not np.isfinite(row["masker_band_db"]) else f"{row['masker_band_db']:.2f}"
        offset = "n/a" if not np.isfinite(row["offset_db"]) else f"{row['offset_db']:.2f}"
        print(f"  {row['probe_hz']:9.0f}  {row['threshold_dbfs']:14.2f}  {band:>14}  {offset:>9}")
    if any(np.isfinite(row["offset_db"]) for row in rows):
        print(
            "\n  offset dB = probe power at threshold relative to the masker's power in that band;\n"
            "  the broadband anchor in the reference document is about -5 dB."
        )


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def export_set(
    directory: Path,
    masker: np.ndarray,
    sample_rate: int,
    rows: list[dict[str, float]],
    offsets: tuple[float, ...],
    seed: int,
    description: str,
    args: argparse.Namespace,
) -> None:
    thresholds = {row["probe_hz"]: row["threshold_dbfs"] for row in rows}
    stimuli, gain = build_blind_set(masker, sample_rate, thresholds, seed=seed, offsets_db=offsets)

    directory.mkdir(parents=True, exist_ok=True)
    write_wav_24(directory / "masker_only.wav", masker * gain, sample_rate)
    for stimulus in stimuli:
        write_wav_24(directory / stimulus.filename, stimulus.signal, sample_rate)

    key = {
        "seed": seed,
        "sample_rate": sample_rate,
        "masker": description,
        "masker_rms": rms_level(masker),
        "global_gain": gain,
        "offsets_db": list(offsets),
        "note": (
            "level_dbfs is the steady probe amplitude after the global gain. "
            "Open this file only after listening."
        ),
        "levels": {
            f"{row['probe_hz']:g}": {
                stimulus.label: round(stimulus.level_dbfs, 2)
                for stimulus in stimuli
                if stimulus.probe_hz == row["probe_hz"]
            }
            for row in rows
        },
        "model_threshold_dbfs": {
            f"{row['probe_hz']:g}": round(row["threshold_dbfs"], 2) for row in rows
        },
    }
    key_path = directory / "answers.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")

    manifest = build_manifest("masker_only.wav", stimuli, sample_rate, description)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nblind listening set written to {directory}")
    print("  masker_only.wav")
    print(f"  f<Hz>_A.wav ... f<Hz>_E.wav  ({len(stimuli)} files)")
    print(f"  manifest.json  (test page reads this; contains no levels)")
    print(f"  {key_path.name}  <- open only after listening")
    if gain != 1.0:
        print(f"  global gain {20 * np.log10(gain):+.2f} dB applied to avoid clipping")
    print(
        "\n  Quiet room, headphones, fixed comfortable volume. For one frequency,\n"
        "  play A..E and note the first label where the steady tone appears.\n"
        "  Then check answers.json: a well-calibrated model puts that boundary at 0 dB."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    freqs = parse_floats(args.freqs, "--freqs")
    offsets = tuple(parse_floats(args.offsets, "--offsets"))
    rng = np.random.default_rng(args.seed)

    masker, sample_rate, description = build_masker(args, rng)
    if sample_rate < MIN_SAMPLE_RATE:
        raise SystemExit(
            f"model needs sample_rate >= {MIN_SAMPLE_RATE} Hz (18 kHz band edge); got {sample_rate} Hz"
        )
    model = PeaqFftModel(sample_rate=sample_rate, playback_db_spl=args.playback_db)
    if masker.size < model.n_fft:
        raise SystemExit(
            f"masker has {masker.size} samples, needs at least one {model.n_fft}-sample frame"
        )

    print(f"masker: {description}")
    print(
        f"  rms {rms_level(masker):.4f} ({20 * np.log10(max(rms_level(masker), 1e-30)):.2f} dBFS)"
        f"  duration {masker.size / sample_rate:.2f} s  {sample_rate} Hz"
    )
    print(
        f"model: playback {args.playback_db:g} dB SPL per full-scale sine "
        "(modelling convention, not real loudness)\n"
    )

    rows: list[dict[str, float]] = []
    silent = rms_level(masker) == 0.0
    for probe_hz in freqs:
        probe = find_probe_threshold(model, masker, probe_hz)
        band_power = band_power_at(model, masker, probe_hz)
        rows.append(
            {
                "probe_hz": probe_hz,
                "threshold_dbfs": probe.threshold_dbfs,
                "threshold_amplitude": probe.threshold_amplitude,
                "masker_band_db": float("nan") if silent else 10.0 * np.log10(max(band_power, 1e-30)),
                "offset_db": (
                    float("nan")
                    if silent
                    else probe_power_offset_db(probe.threshold_amplitude, band_power)
                ),
            }
        )
        if probe.threshold_dbfs > QUIET_SWEEP_CEILING_DBFS:
            print(
                f"  warning: threshold at {probe_hz:g} Hz hit the 0 dBFS sweep ceiling; "
                "raise the masker level or lower the probe range"
            )

    print_table(rows)

    if args.export is not None:
        export_set(args.export, masker, sample_rate, rows, offsets, args.seed, description, args)

    if args.json is not None:
        payload = {
            "masker": description,
            "sample_rate": sample_rate,
            "duration_s": masker.size / sample_rate,
            "masker_rms": rms_level(masker),
            "playback_db_spl_assumption": args.playback_db,
            "seed": args.seed,
            "score_version": "peaq-fft-masking-mvp-v1",
            "thresholds": [
                {key: finite_or_none(value) for key, value in row.items()} for row in rows
            ],
            "disclaimer": (
                "Model prediction in the digital domain; not a measured human "
                "threshold and not a claim about real playback loudness."
            ),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nthreshold table written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
