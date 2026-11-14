"""Tests for the blind-listening stimulus builder."""

import json

import numpy as np
import pytest

from a2t.evaluation.listening import (
    LISTENING_LABELS,
    LISTENING_OFFSETS_DB,
    build_blind_set,
    build_manifest,
    faded_tone,
)

SAMPLE_RATE = 48000
N_SAMPLES = 4800


def make_masker() -> np.ndarray:
    rng = np.random.default_rng(3)
    return 0.05 * rng.standard_normal(N_SAMPLES)


def test_labels_cover_offsets_per_frequency() -> None:
    thresholds = {1000.0: -60.0, 2000.0: -55.0}
    stimuli, gain = build_blind_set(make_masker(), SAMPLE_RATE, thresholds, seed=1)

    assert gain == 1.0
    assert len(stimuli) == len(thresholds) * len(LISTENING_OFFSETS_DB)
    for probe_hz, threshold in thresholds.items():
        entries = [s for s in stimuli if s.probe_hz == probe_hz]
        assert {s.label for s in entries} == set(LISTENING_LABELS)
        assert sorted(s.level_dbfs for s in entries) == pytest.approx(
            sorted(threshold + offset for offset in LISTENING_OFFSETS_DB)
        )
        for stimulus in entries:
            assert stimulus.signal.size == N_SAMPLES
            assert stimulus.filename == f"f{probe_hz:g}_{stimulus.label}.wav"


def test_same_seed_is_deterministic() -> None:
    thresholds = {1000.0: -60.0}
    first, gain_first = build_blind_set(make_masker(), SAMPLE_RATE, thresholds, seed=7)
    second, gain_second = build_blind_set(make_masker(), SAMPLE_RATE, thresholds, seed=7)

    assert gain_first == gain_second
    assert [s.label for s in first] == [s.label for s in second]
    assert [s.level_dbfs for s in first] == [s.level_dbfs for s in second]
    for left, right in zip(first, second):
        assert np.array_equal(left.signal, right.signal)


def test_gain_keeps_peak_under_limit() -> None:
    masker = np.full(1000, 0.95)
    stimuli, gain = build_blind_set(
        masker, SAMPLE_RATE, {1000.0: -6.0}, seed=0, offsets_db=(0.0,)
    )

    assert gain < 1.0
    assert np.max(np.abs(stimuli[0].signal)) <= 0.999 + 1e-12
    assert stimuli[0].level_dbfs < -6.0


def test_faded_tone_starts_and_ends_at_zero() -> None:
    tone = faded_tone(1000.0, 0.5, N_SAMPLES, SAMPLE_RATE)

    assert abs(tone[0]) < 1e-12
    assert abs(tone[-1]) < 1e-12
    assert np.max(np.abs(tone)) <= 0.5 + 1e-12


def test_faded_tone_rejects_out_of_range_frequency() -> None:
    with pytest.raises(ValueError):
        faded_tone(0.0, 0.5, N_SAMPLES, SAMPLE_RATE)
    with pytest.raises(ValueError):
        faded_tone(SAMPLE_RATE / 2.0, 0.5, N_SAMPLES, SAMPLE_RATE)


def test_rejects_empty_thresholds() -> None:
    with pytest.raises(ValueError):
        build_blind_set(make_masker(), SAMPLE_RATE, {}, seed=0)


def test_manifest_lists_files_without_levels() -> None:
    thresholds = {1000.0: -60.0, 2000.0: -55.0}
    stimuli, _ = build_blind_set(make_masker(), SAMPLE_RATE, thresholds, seed=2)
    manifest = build_manifest("masker_only.wav", stimuli, SAMPLE_RATE, "pink noise")

    assert manifest["masker_only"] == "masker_only.wav"
    assert manifest["masker"] == "pink noise"
    assert manifest["sample_rate"] == SAMPLE_RATE
    assert [f["hz"] for f in manifest["frequencies"]] == [1000.0, 2000.0]
    for freq in manifest["frequencies"]:
        assert len(freq["stimuli"]) == len(LISTENING_LABELS)
        for stimulus in freq["stimuli"]:
            assert set(stimulus) == {"file", "label"}

    text = json.dumps(manifest)
    assert "level" not in text
    assert "dBFS" not in text


def test_manifest_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError):
        build_manifest("masker_only.wav", [], 0)


def test_exported_stimuli_carry_the_labelled_level(tmp_path) -> None:
    from a2t.data.wav import read_wav_mono, write_wav_24

    rng = np.random.default_rng(11)
    masker = 0.05 * rng.standard_normal(SAMPLE_RATE)  # 1 s keeps fades negligible
    stimuli, gain = build_blind_set(masker, SAMPLE_RATE, {1000.0: -60.0}, seed=5)

    write_wav_24(tmp_path / "masker_only.wav", masker * gain, SAMPLE_RATE)
    for stimulus in stimuli:
        write_wav_24(tmp_path / stimulus.filename, stimulus.signal, SAMPLE_RATE)

    loaded_masker, _ = read_wav_mono(tmp_path / "masker_only.wav")
    middle = slice(loaded_masker.size // 4, 3 * loaded_masker.size // 4)
    for stimulus in stimuli:
        loaded, _ = read_wav_mono(tmp_path / stimulus.filename)
        residual = loaded[middle] - loaded_masker[middle]
        measured = 20.0 * np.log10(np.sqrt(np.mean(residual**2)) * np.sqrt(2.0))
        assert measured == pytest.approx(stimulus.level_dbfs, abs=0.01)
