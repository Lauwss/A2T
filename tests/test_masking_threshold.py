"""Behavioural and anchor-validation tests for the masking-threshold evaluator.

Tolerances come from the measured results recorded in
``docs/evaluation/MASKING_THRESHOLD_MVP.md`` and
``artifacts/masking_mvp/validation_results.json`` (seed-fixed, deterministic).
"""

import numpy as np
import pytest

from a2t.evaluation import (
    EvaluationInput,
    MaskingEvaluator,
    MaskingThresholdEvaluator,
    compare_conditions,
    find_probe_threshold,
    narrowband_noise,
    pink_noise,
)
from a2t.evaluation.psychoacoustics import PeaqFftModel, ear_weight_db

SAMPLE_RATE = 48000
N_SAMPLES = 48000
SEED = 20261025


def make_target(burst_amplitude: float = 0.05) -> np.ndarray:
    """Two tone bursts separated by silence: two events, one tonal band each."""
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    target = np.zeros(N_SAMPLES)
    burst1 = (t >= 0.2) & (t < 0.5)
    burst2 = (t >= 0.8) & (t < 1.1)
    target[burst1] = burst_amplitude * np.sin(2 * np.pi * 1000.0 * t[burst1])
    target[burst2] = burst_amplitude * np.sin(2 * np.pi * 2000.0 * t[burst2])
    return target


def make_envelope_masker(rms: float) -> np.ndarray:
    """Amplitude-modulated pink noise (rain-like slow envelope)."""
    rng = np.random.default_rng(SEED)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    envelope = 0.75 + 0.25 * np.sin(2 * np.pi * 0.5 * t)
    noise = pink_noise(N_SAMPLES, rms, SAMPLE_RATE, rng)
    scaled = noise * envelope
    return scaled * (rms / np.sqrt(np.mean(scaled * scaled)))


def test_report_shapes_and_levels() -> None:
    target = make_target()
    masker = make_envelope_masker(0.1)
    report = MaskingThresholdEvaluator().score(
        EvaluationInput(target, masker, target + masker, SAMPLE_RATE)
    )
    n_bands = len(report.band_center_hz)
    assert report.margins_target_db.shape == (n_bands, report.n_frames)
    assert report.margins_noise_db.shape == (n_bands, report.n_frames)
    assert report.frame_active.shape == (report.n_frames,)
    assert 0.0 <= report.masked_frame_ratio <= 1.0
    assert report.score_version == "peaq-fft-masking-mvp-v1"
    assert report.sample_rate == SAMPLE_RATE
    assert report.frame_hop_samples == 1024


def test_silent_target_has_no_events_and_no_masked_frames() -> None:
    masker = make_envelope_masker(0.1)
    report = MaskingThresholdEvaluator().score(
        EvaluationInput(np.zeros(N_SAMPLES), masker, masker, SAMPLE_RATE)
    )
    assert report.events == ()
    assert report.masked_frame_ratio == 0.0
    assert report.mean_active_margin_db == 0.0
    assert not np.any(np.isnan(report.margins_target_db))


def test_unmasked_loud_target_is_fully_audible() -> None:
    target = make_target()
    silence = np.zeros(N_SAMPLES)
    report = MaskingThresholdEvaluator().score(
        EvaluationInput(target, silence, target, SAMPLE_RATE)
    )
    assert report.frame_active.any()
    assert report.masked_frame_ratio == 0.0
    assert np.all(report.frame_audible[report.frame_active])


def test_two_bursts_produce_two_events() -> None:
    target = make_target()
    masker = make_envelope_masker(0.1)
    report = MaskingThresholdEvaluator().score(
        EvaluationInput(target, masker, target + masker, SAMPLE_RATE)
    )
    assert len(report.events) == 2
    first, second = report.events
    assert first.end_frame <= second.start_frame
    for event in report.events:
        assert 0.0 <= event.masked_ratio <= 1.0
        assert np.isfinite(event.min_margin_db)


def test_increasing_masker_level_reduces_margin() -> None:
    target = make_target()
    quiet_masker = make_envelope_masker(0.032)
    loud_masker = make_envelope_masker(0.1)
    quiet = MaskingThresholdEvaluator().score(
        EvaluationInput(target, quiet_masker, target + quiet_masker, SAMPLE_RATE)
    )
    loud = MaskingThresholdEvaluator().score(
        EvaluationInput(target, loud_masker, target + loud_masker, SAMPLE_RATE)
    )
    delta = quiet.mean_active_margin_db - loud.mean_active_margin_db
    # 9.9 dB masker increase -> ~10.6 dB margin drop (measured growth 1.067 dB/dB).
    assert 8.0 <= delta <= 13.0
    assert loud.masked_frame_ratio >= quiet.masked_frame_ratio


def test_mismatched_pairing_masks_less_than_correct_pairing() -> None:
    # Burst level sits near the masking threshold of the 0.1 RMS masker
    # (measured broadband probe threshold is about -46 dBFS), so alignment
    # quality is visible in the margins.
    target = make_target(burst_amplitude=0.006)
    masker = make_envelope_masker(0.1)
    correct = MaskingThresholdEvaluator().score(
        EvaluationInput(target, masker, target + masker, SAMPLE_RATE)
    )
    rolled = np.roll(masker, masker.size // 2)
    mismatched = MaskingThresholdEvaluator().score(
        EvaluationInput(target, rolled, target + rolled, SAMPLE_RATE)
    )
    assert correct.mean_active_margin_db < mismatched.mean_active_margin_db


def test_broadband_masking_offset_matches_reference_anchor() -> None:
    """Anchor: noise masks a tone about 5 dB below its in-band power (doc 3.5)."""
    model = PeaqFftModel(sample_rate=SAMPLE_RATE)
    rng = np.random.default_rng(SEED)
    masker = pink_noise(N_SAMPLES, 0.1, SAMPLE_RATE, rng)
    probe = find_probe_threshold(model, masker, 1000.0)
    band = int(np.argmin(np.abs(model.center_hz - 1000.0)))
    weight = 10.0 ** (float(ear_weight_db(model.center_hz[band])) / 10.0)
    energy = model.band_energy(model.spectrum(masker))
    band_power = float(energy[band].mean()) / (model.fac**2 * weight)
    offset = 10.0 * np.log10((probe.threshold_amplitude**2 / 2.0) / band_power)
    assert offset == pytest.approx(-5.0, abs=3.0)


def test_quiet_threshold_matches_loudness_model_anchor() -> None:
    """Anchor: threshold in quiet near eq (60), within +-6 dB SPL-equivalent."""
    model = PeaqFftModel(sample_rate=SAMPLE_RATE)
    for freq, expected_delta in ((1000.0, 0.81), (2000.0, -0.10), (4000.0, -0.11)):
        probe = find_probe_threshold(model, np.zeros(N_SAMPLES), freq)
        spl = 92.0 + probe.threshold_dbfs + float(ear_weight_db(freq))
        anchor = 0.364 * (freq / 1000.0) ** -0.8
        assert spl - anchor == pytest.approx(expected_delta, abs=3.0)


def test_narrowband_masker_offset_is_documented_limitation() -> None:
    """Narrowband maskers over-mask by ~10 dB; locked so regressions surface."""
    model = PeaqFftModel(sample_rate=SAMPLE_RATE)
    rng = np.random.default_rng(SEED + 1)
    masker = narrowband_noise(N_SAMPLES, 0.1, SAMPLE_RATE, 1000.0, 100.0, rng)
    probe = find_probe_threshold(model, masker, 1000.0)
    band = int(np.argmin(np.abs(model.center_hz - 1000.0)))
    weight = 10.0 ** (float(ear_weight_db(model.center_hz[band])) / 10.0)
    energy = model.band_energy(model.spectrum(masker))
    band_power = float(energy[band].mean()) / (model.fac**2 * weight)
    offset = 10.0 * np.log10((probe.threshold_amplitude**2 / 2.0) / band_power)
    assert -25.0 <= offset <= -5.0


def test_compare_conditions_returns_four_deterministic_controls() -> None:
    target = make_target()
    masker = make_envelope_masker(0.1)
    first = compare_conditions(target, masker, SAMPLE_RATE, seed=SEED)
    second = compare_conditions(target, masker, SAMPLE_RATE, seed=SEED)
    assert set(first.reports) == {
        "correct",
        "mismatched",
        "unconditional",
        "traditional_pink",
    }
    summary = first.summary()
    for name, values in summary.items():
        assert 0.0 <= values["masked_frame_ratio"] <= 1.0
        assert np.isfinite(values["nmr_total_db"])
    # Deterministic under a fixed seed.
    assert summary == second.summary()
    # The unconditional control cannot mask anything.
    assert first.reports["unconditional"].masked_frame_ratio == 0.0
    # The correctly aligned modulated masker hides more than the traditional
    # level-matched pink baseline on the tonal bursts.
    assert (
        first.reports["correct"].mean_active_margin_db
        <= first.reports["traditional_pink"].mean_active_margin_db
    )


def test_nmr_aggregates_move_with_target_level() -> None:
    target = make_target()
    masker = make_envelope_masker(0.1)
    loud = MaskingThresholdEvaluator().score(
        EvaluationInput(target, masker, target + masker, SAMPLE_RATE)
    )
    quiet = MaskingThresholdEvaluator().score(
        EvaluationInput(0.05 * target, masker, 0.05 * target + masker, SAMPLE_RATE)
    )
    assert loud.nmr_total_db > quiet.nmr_total_db


def test_invalid_inputs_are_rejected() -> None:
    evaluator = MaskingThresholdEvaluator()
    good = np.zeros(48000)
    with pytest.raises(ValueError):
        evaluator.score(EvaluationInput(good[:-1], good, good, SAMPLE_RATE))
    with pytest.raises(ValueError):
        evaluator.score(EvaluationInput(np.zeros(4), good, good, SAMPLE_RATE))
    with pytest.raises(ValueError):
        evaluator.score(
            EvaluationInput(np.zeros(48000), good, good, 8000)
        )


def test_legacy_evaluator_unchanged() -> None:
    frame = [1.0] * 8
    scores = MaskingEvaluator(frame_size=8, hop_size=8).score(
        EvaluationInput(frame, [2.0] * 8, [3.0] * 8, 24000)
    )
    assert len(scores.threshold_exposure_db) == 1
    assert scores.threshold_exposure_db[0] < 0
