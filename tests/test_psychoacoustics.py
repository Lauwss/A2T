"""Unit tests for the PEAQ FFT ear-model frontend.

Reference values are taken from the project reference document
(ITU-R BS.1387-2 Chinese translation: Table 6 band edges, eq 2/7/13/60) and
from hand-evaluated constants of those equations.
"""

import numpy as np
import pytest

from a2t.evaluation.psychoacoustics import (
    BARK_RES,
    N_FFT,
    PeaqFftModel,
    bark_to_hz,
    critical_band_edges,
    ear_weight_db,
    hz_to_bark,
    peaq_window,
)

# Table 6 of the reference document: (band, lower_hz, center_hz, upper_hz)
TABLE6_ROWS = [
    (0, 80.0, 91.708, 103.445),
    (1, 103.445, 115.216, 127.023),
    (30, 974.336, 995.408, 1016.797),
    (54, 2486.169, 2532.456, 2579.551),
    (107, 16775.035, 17077.504, 17385.42),
    (108, 17385.42, 17690.045, 18000.0),
]


def test_critical_band_count_is_109() -> None:
    lower, center, upper = critical_band_edges()
    assert lower.size == center.size == upper.size == 109


@pytest.mark.parametrize("band,lower,center,upper", TABLE6_ROWS)
def test_critical_band_edges_match_table6(
    band: int, lower: float, center: float, upper: float
) -> None:
    lower_hz, center_hz, upper_hz = critical_band_edges()
    assert lower_hz[band] == pytest.approx(lower, abs=1.0)
    # The last band's center follows from the capped upper edge, so its table
    # value differs from the Bark-domain midpoint by a few Hz.
    center_tol = 15.0 if band == 108 else 2.0
    assert center_hz[band] == pytest.approx(center, abs=center_tol)
    assert upper_hz[band] == pytest.approx(upper, abs=2.0)


def test_last_band_upper_edge_capped_at_18khz() -> None:
    _, _, upper = critical_band_edges()
    assert upper[-1] == 18000.0
    assert upper[0] == pytest.approx(103.445, abs=1.0)


def test_bark_scale_round_trip() -> None:
    assert float(hz_to_bark(1000.0)) == pytest.approx(8.51, abs=0.05)
    for freq in (80.0, 500.0, 1000.0, 4000.0, 18000.0):
        assert float(bark_to_hz(hz_to_bark(freq))) == pytest.approx(freq, rel=1e-9)


def test_window_properties() -> None:
    window = peaq_window()
    assert window[0] == pytest.approx(0.0, abs=1e-12)
    # The N-1 denominator puts the exact cos=-1 point between samples, so the
    # midpoint only reaches the peak within a few parts per million.
    assert window[N_FFT // 2] == pytest.approx(np.sqrt(8.0 / 3.0), abs=1e-4)
    # Sum of (1 - cos(2*pi*k/(N-1))) over k = 0..N-1 equals N-1 exactly, since
    # the k = N-1 term repeats cos(2*pi) = 1.
    assert window.sum() == pytest.approx(0.5 * np.sqrt(8.0 / 3.0) * (N_FFT - 1), rel=1e-6)


@pytest.mark.parametrize(
    "freq,expected", [(1000.0, -1.9129), (4000.0, 3.9733), (80.0, -16.4604)]
)
def test_ear_weight_matches_eq7(freq: float, expected: float) -> None:
    assert float(ear_weight_db(freq)) == pytest.approx(expected, abs=0.02)


def test_internal_noise_matches_eq13() -> None:
    model = PeaqFftModel(sample_rate=48000)
    band_1k = int(np.argmin(np.abs(model.center_hz - 1000.0)))
    assert model.p_threshold[band_1k] == pytest.approx(10.0 ** (0.4 * 0.364), rel=0.02)
    assert model.absolute_threshold[band_1k] == pytest.approx(10.0**0.364, rel=0.02)


def test_masking_offset_is_continuous_and_rising() -> None:
    model = PeaqFftModel(sample_rate=48000, masking_offset_calibration_db=0.0)
    offset_db = 10.0 * np.log10(model.mask_offset)
    boundary = int(round(12.0 / BARK_RES))  # last band at or below 12 Bark
    assert offset_db[: boundary + 1] == pytest.approx(3.0)
    # Above 12 Bark the offset rises with the band index.
    assert np.all(np.diff(offset_db[boundary:]) > 0.0)
    assert offset_db[-1] == pytest.approx(0.25 * (model.n_bands - 1) * BARK_RES, abs=0.1)


def test_calibration_maps_full_scale_sine_to_playback_level() -> None:
    model = PeaqFftModel(sample_rate=48000, playback_db_spl=92.0)
    n = model.sample_rate  # 1 s of 1019.5 Hz full-scale sine
    t = np.arange(n) / model.sample_rate
    sine = np.sin(2.0 * np.pi * 1019.5 * t)
    energy = model.band_energy(model.spectrum(sine))
    band = int(np.argmin(np.abs(model.center_hz - 1019.5)))
    level_db = 10.0 * np.log10(float(energy[band].max()))
    weight_db = float(ear_weight_db(model.center_hz[band]))
    assert level_db == pytest.approx(92.0 + weight_db, abs=2.0)


def test_spread_preserves_flat_pattern_at_zero_level() -> None:
    model = PeaqFftModel(sample_rate=48000)
    flat = np.ones((model.n_bands, 3))
    result = model.spread(flat)
    assert np.allclose(result, flat, rtol=1e-6)


def test_spread_of_localized_pattern_decays_both_ways() -> None:
    model = PeaqFftModel(sample_rate=48000)
    pattern = np.zeros((model.n_bands, 1))
    center = 50
    pattern[center, 0] = 1e6
    result = model.spread(pattern + 1e-12)[:, 0]
    up = result[center + 2]
    down = result[center - 2]
    assert result[center] > up > 0.0
    assert result[center] > down > 0.0
    # Upward spread reaches farther than downward spread (level-dependent
    # upper slope shallower than the fixed 27 dB/Bark lower slope).
    assert up > down


def test_masking_threshold_in_quiet_equals_absolute_threshold() -> None:
    model = PeaqFftModel(sample_rate=48000)
    threshold = model.masking_threshold(np.zeros(model.sample_rate))
    assert np.allclose(threshold, model.absolute_threshold[:, None])


def test_model_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        PeaqFftModel(sample_rate=16000)
    with pytest.raises(ValueError):
        PeaqFftModel(playback_db_spl=0.0)
