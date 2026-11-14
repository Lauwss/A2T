"""Round-trip and downmix tests for the dependency-free WAV I/O."""

import wave

import numpy as np
import pytest

from a2t.data.wav import read_wav_mono, write_wav_24


def test_roundtrip_24bit(tmp_path) -> None:
    rng = np.random.default_rng(0)
    signal = np.clip(0.3 * rng.standard_normal(1000), -1.0, 1.0)
    path = tmp_path / "roundtrip.wav"

    write_wav_24(path, signal, 48000)
    loaded, sample_rate = read_wav_mono(path)

    assert sample_rate == 48000
    assert loaded.shape == signal.shape
    assert np.max(np.abs(loaded - signal)) <= 1.5 * 2.0**-23


def test_stereo_downmix(tmp_path) -> None:
    path = tmp_path / "stereo.wav"
    left = np.full(200, 0.5)
    right = np.full(200, 0.25)
    interleaved = np.empty(400, dtype="<i2")
    interleaved[0::2] = np.round(left * 32767).astype("<i2")
    interleaved[1::2] = np.round(right * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(interleaved.tobytes())

    loaded, sample_rate = read_wav_mono(path)

    assert sample_rate == 44100
    assert loaded.size == 200
    assert np.allclose(loaded, 0.375, atol=1e-4)


def test_write_clips_above_full_scale(tmp_path) -> None:
    path = tmp_path / "clipped.wav"
    signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    write_wav_24(path, signal, 48000)
    loaded, _ = read_wav_mono(path)

    assert loaded[0] == pytest.approx(-1.0)
    assert loaded[-1] == pytest.approx(1.0 - 2.0**-23, rel=0, abs=1e-9)


def test_rejects_bad_inputs(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_wav_24(tmp_path / "empty.wav", np.array([]), 48000)
    with pytest.raises(ValueError):
        write_wav_24(tmp_path / "2d.wav", np.zeros((2, 2)), 48000)
    with pytest.raises(ValueError):
        write_wav_24(tmp_path / "nan.wav", np.array([0.0, np.nan]), 48000)
    with pytest.raises(FileNotFoundError):
        read_wav_mono(tmp_path / "missing.wav")
