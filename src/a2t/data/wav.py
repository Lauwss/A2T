"""Minimal PCM WAV I/O on top of the standard library.

Only what the evaluation tooling needs: read any integer-PCM WAV into a mono
float array, and write 24-bit mono PCM without extra dependencies. Source files
are opened read-only; nothing is ever written back to the input path.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SUPPORTED_SAMPLE_WIDTHS = (1, 2, 3, 4)
FULL_SCALE_24 = float(1 << 23)


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an integer-PCM WAV file and downmix it to mono float64 in [-1, 1)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise ValueError(
            f"unsupported sample width {sample_width * 8} bit in {path}; "
            "expected 8, 16, 24, or 32 bit integer PCM"
        )
    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / float(1 << 15)
    elif sample_width == 3:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        ints = np.where(ints >= (1 << 23), ints - (1 << 24), ints)
        data = ints.astype(np.float64) / FULL_SCALE_24
    else:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, int(sample_rate)


def write_wav_24(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    """Write a mono 24-bit PCM WAV, clipping to full scale."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must be finite")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    clipped = np.clip(signal, -1.0, 1.0 - 2.0**-23)
    ints = np.round(clipped * FULL_SCALE_24).astype(np.int64)
    ints = np.clip(ints, -(1 << 23), (1 << 23) - 1).astype("<i4")
    payload = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(sample_rate)
        wf.writeframes(payload)
