"""PEAQ FFT ear-model frontend for masking-threshold estimation.

Implements the FFT-based peripheral ear model of ITU-R BS.1387-2 as transcribed
in ``docs/感知音频质量的客观测量方法/感知音频质量的客观测量方法_后附资料4及附件2.md``
(附件2 §2.1): framing (eq 1), Hann window (eq 2), DFT (eq 4), playback-level
scaling (eq 5-6), outer/middle-ear weighting (eq 7), Schroeder pitch scale
(eq 10), critical-band grouping (eq 11 plus the annex pseudocode), internal
noise (eq 13), level-dependent frequency spreading (eq 15-20), temporal
spreading for forward masking (eq 21-24) and the masking pattern (eq 25-26).

Level convention: a full-scale (0 dBFS) sine wave corresponds to the assumed
playback sound pressure level ``playback_db_spl`` (the source document
recommends 92 dB SPL when the playback level is unknown). Everything is
reported in the digital domain; no real-ear SPL claim is made.

Known deviations from the literal source text, all recorded in
``docs/evaluation/MASKING_THRESHOLD_MVP.md``:

- ``Norm`` in eq (5) is implemented as the reciprocal of the maximum spectral
  magnitude of the 1019.5 Hz full-scale calibration sine, so that the
  calibration tone maps exactly to ``playback_db_spl``. Reading ``Norm`` as the
  raw maximum would place the same tone ~18 dB lower and contradict the stated
  purpose of eq (5-6).
- The last critical band's upper edge is capped at 18 kHz (Table 6 behaviour).
- The level-dependent upper spreading slope is clamped to at most -4 dB/Bark,
  following eq (33) of the filter-bank model, to keep the kernel decaying for
  very high band levels.
"""

from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 1024
F_LOW_HZ = 80.0
F_HIGH_HZ = 18000.0
BARK_RES = 0.25
CALIBRATION_HZ = 1019.5
CALIBRATION_FRAMES = 10
ENERGY_FLOOR = 1e-12
UPPER_SLOPE_CLAMP_DB = -4.0
LOWER_SLOPE_DB_PER_BARK = 27.0
TAU_MIN_S = 0.008
TAU_100HZ_S = 0.030


def hz_to_bark(f_hz: float | np.ndarray) -> float | np.ndarray:
    """Schroeder pitch approximation, eq (10) of the source document."""
    return 7.0 * np.arcsinh(np.asarray(f_hz, dtype=float) / 650.0)


def bark_to_hz(z_bark: float | np.ndarray) -> float | np.ndarray:
    """Inverse of :func:`hz_to_bark`."""
    return 650.0 * np.sinh(np.asarray(z_bark, dtype=float) / 7.0)


def critical_band_edges(
    res: float = BARK_RES,
    f_low_hz: float = F_LOW_HZ,
    f_high_hz: float = F_HIGH_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lower_hz, center_hz, upper_hz)`` for equal-spacing Bark bands.

    Band edges sit on a uniform grid of ``res`` Bark between ``f_low_hz`` and
    ``f_high_hz``; centers are midpoints in the Bark domain. This reproduces
    Table 6 of the source document (109 bands at 0.25 Bark resolution) with the
    last band's upper edge capped at ``f_high_hz``.
    """
    z_low = float(hz_to_bark(f_low_hz))
    z_high = float(hz_to_bark(f_high_hz))
    n_bands = int(np.ceil((z_high - z_low) / res - 1e-9))
    grid = z_low + res * np.arange(n_bands + 1)
    lower = bark_to_hz(grid[:-1])
    upper = bark_to_hz(grid[1:])
    center = bark_to_hz(grid[:-1] + res / 2.0)
    upper[-1] = f_high_hz
    center[-1] = 0.5 * (lower[-1] + upper[-1])
    return lower, center, upper


def peaq_window(n_fft: int = N_FFT) -> np.ndarray:
    """Hann-family analysis window, eq (2) of the source document."""
    k = np.arange(n_fft, dtype=float)
    return 0.5 * np.sqrt(8.0 / 3.0) * (1.0 - np.cos(2.0 * np.pi * k / (n_fft - 1)))


def ear_weight_db(f_hz: float | np.ndarray) -> float | np.ndarray:
    """Outer/middle-ear weighting in dB, eq (7) of the source document."""
    f_khz = np.asarray(f_hz, dtype=float) / 1000.0
    weight = (
        -0.6 * 3.64 * np.power(f_khz, -0.8, where=f_khz > 0, out=np.full_like(f_khz, np.nan))
        + 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
        - 1e-3 * np.power(f_khz, 3.6)
    )
    return np.where(f_khz > 0, weight, -200.0)


class PeaqFftModel:
    """FFT ear model mapping a signal to band-domain patterns and thresholds.

    Arrays are shaped ``(n_bands, n_frames)`` with bands on the 0.25 Bark grid
    and frames advanced by :data:`HOP` samples (eq 1).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        playback_db_spl: float = 92.0,
        n_fft: int = N_FFT,
        hop: int = HOP,
        masking_offset_calibration_db: float = 8.3,
    ) -> None:
        if sample_rate < 2.0 * F_HIGH_HZ:
            raise ValueError(
                f"sample_rate must cover the {F_HIGH_HZ:.0f} Hz band edge "
                f"(got {sample_rate} Hz)"
            )
        if playback_db_spl <= 0:
            raise ValueError("playback_db_spl must be positive")
        if n_fft <= 0 or hop <= 0 or hop > n_fft:
            raise ValueError("require 0 < hop <= n_fft")

        self.sample_rate = int(sample_rate)
        self.playback_db_spl = float(playback_db_spl)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.bin_hz = self.sample_rate / self.n_fft
        self.window = peaq_window(n_fft)

        self.lower_hz, self.center_hz, self.upper_hz = critical_band_edges()
        self.n_bands = int(self.center_hz.size)

        # Critical-band grouping fractions per (band, bin); equivalent to the
        # four overlap cases of the annex pseudocode.
        bin_index = np.arange(self.n_fft // 2)
        bin_left = (bin_index - 0.5) * self.bin_hz
        bin_right = (bin_index + 0.5) * self.bin_hz
        overlap = (
            np.minimum(bin_right[None, :], self.upper_hz[:, None])
            - np.maximum(bin_left[None, :], self.lower_hz[:, None])
        )
        self.bin_frac = np.clip(overlap, 0.0, None) / self.bin_hz

        # Internal noise floor added to band energies, eq (13).
        fc_khz = self.center_hz / 1000.0
        self.p_threshold = 10.0 ** (0.4 * 0.364 * np.power(fc_khz, -0.8))
        # Loudness-model threshold in quiet, eq (60); used as the absolute
        # floor of the masking threshold (detection in quiet is a different
        # mechanism from masking, and the calibrated masking offset must not
        # lower the threshold in silence).
        self.absolute_threshold = 10.0 ** (0.364 * np.power(fc_khz, -0.8))

        # Masking offsets, eq (25)-(26). The literal transcription "E/10^(10/m)"
        # implies a 33.3 dB offset that contradicts the same document's
        # section 3.5 anchor (noise masks a tone about 5 dB below its level) and
        # the absolute-threshold anchor; we read it as the numerator/denominator
        # swap "E/10^(m/10)", i.e. an m[k]-dB offset: 3.0 dB up to 12 Bark, then
        # m = 0.25*k*res rising to 6.75 dB. On top of that,
        # ``masking_offset_calibration_db`` is a single global constant derived
        # from the section 3.5 anchor with broadband (pink) maskers: without it
        # the model under-predicts masking depth by ~8 dB. See the design doc.
        k_bark = np.arange(self.n_bands) * BARK_RES
        m_weight = np.where(k_bark <= 12.0, 3.0, 0.25 * k_bark)
        total_offset_db = m_weight + masking_offset_calibration_db
        self.mask_offset = 10.0 ** (total_offset_db / 10.0)

        # Level-independent normalisation of the spreading p-norm, eq (19)-(20),
        # evaluated with slopes at L = 0.
        su_zero = -24.0 - 230.0 / self.center_hz
        kernel0 = self._kernel(su_zero[:, None] * np.ones((self.n_bands, self.n_bands)))
        etilde = kernel0 / kernel0.sum(axis=1, keepdims=True)
        self.norm_sp = np.power(np.power(etilde, 0.4).sum(axis=0), 1.0 / 0.4)

        self.norm = self._calibrate_norm()
        self.fac = self.norm * (10.0 ** (self.playback_db_spl / 20.0))

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------
    def _frames(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError("input signal must be one-dimensional")
        if not np.all(np.isfinite(x)):
            raise ValueError("input signal must be finite")
        if x.size < self.n_fft:
            raise ValueError(
                f"signal needs at least {self.n_fft} samples for one frame"
            )
        n_frames = 1 + (x.size - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        return x[idx]

    def spectrum(self, x: np.ndarray) -> np.ndarray:
        """Ear-weighted amplitude spectra ``Fe`` with shape (n_frames, n_bins).

        Applies the analysis window (eq 2-3), the 1/N DFT (eq 4), the playback
        scaling (eq 5-6) and the ear weighting (eq 7-9).
        """
        frames = self._frames(np.asarray(x, dtype=float))
        windowed = frames * self.window[None, :]
        spectrum = np.fft.fft(windowed, n=self.n_fft, axis=1) / self.n_fft
        one_sided = np.abs(spectrum[:, : self.n_fft // 2]) * self.fac
        weight = 10.0 ** (
            np.asarray(ear_weight_db(np.arange(self.n_fft // 2) * self.bin_hz)) / 20.0
        )
        return one_sided * weight[None, :]

    def band_energy(self, ear_weighted: np.ndarray) -> np.ndarray:
        """Group bin energies into critical bands, eq (11) plus pseudocode.

        ``ear_weighted`` has shape (n_frames, n_bins); returns (n_bands, n_frames).
        """
        energy = (ear_weighted**2) @ self.bin_frac.T
        return np.maximum(energy.T, ENERGY_FLOOR)

    # ------------------------------------------------------------------
    # Pattern stages
    # ------------------------------------------------------------------
    def pitch_pattern(self, x: np.ndarray) -> np.ndarray:
        """Pitch pattern ``Pp`` with internal noise added, eq (11)-(14)."""
        return self.band_energy(self.spectrum(x)) + self.p_threshold[:, None]

    def _kernel(self, su: np.ndarray) -> np.ndarray:
        """Spreading kernel rows per source band for upper slopes ``su`` (dB/Bark)."""
        offset = (np.arange(self.n_bands)[None, :] - np.arange(self.n_bands)[:, None]) * BARK_RES
        slope = np.where(offset >= 0.0, su, LOWER_SLOPE_DB_PER_BARK)
        return np.power(10.0, offset * slope / 10.0)

    def spread(self, pitch_pattern: np.ndarray) -> np.ndarray:
        """Level-dependent frequency spreading, eq (15)-(20)."""
        level = 10.0 * np.log10(pitch_pattern)
        su = np.minimum(-24.0 - 230.0 / self.center_hz[:, None] + 0.2 * level, UPPER_SLOPE_CLAMP_DB)
        result = np.empty_like(pitch_pattern)
        power_pattern = np.power(pitch_pattern, 0.4)
        for n_frame in range(pitch_pattern.shape[1]):
            kernel = self._kernel(su[:, n_frame][:, None] * np.ones((1, self.n_bands)))
            kernel_norm = kernel / kernel.sum(axis=1, keepdims=True)
            contributions = power_pattern[:, n_frame][:, None] * np.power(kernel_norm, 0.4)
            result[:, n_frame] = np.power(contributions.sum(axis=0), 1.0 / 0.4) / self.norm_sp
        return result

    def temporal_spread(self, spread_pattern: np.ndarray) -> np.ndarray:
        """Forward-masking time smoothing, eq (21)-(24)."""
        tau = TAU_MIN_S + (100.0 / self.center_hz) * (TAU_100HZ_S - TAU_MIN_S)
        decay = np.exp(-((self.hop / self.sample_rate) / tau))[:, None]
        smoothed = np.zeros_like(spread_pattern)
        result = np.empty_like(spread_pattern)
        for n_frame in range(spread_pattern.shape[1]):
            current = spread_pattern[:, n_frame : n_frame + 1]
            smoothed = decay * smoothed + (1.0 - decay) * current
            result[:, n_frame] = np.maximum(smoothed[:, 0], current[:, 0])
        return result

    def excitation(self, x: np.ndarray) -> np.ndarray:
        """Excitation pattern ``E``: pitch pattern, spread, time-smoothed."""
        return self.temporal_spread(self.spread(self.pitch_pattern(x)))

    def masking_threshold(self, x: np.ndarray) -> np.ndarray:
        """Masking pattern ``M`` derived from the signal's excitation, eq (25)-(26).

        The threshold is the maximum of the excitation-derived masking
        threshold and the loudness-model absolute threshold (eq 60): masking
        never makes a signal harder to detect than threshold in quiet.
        """
        masked = self.excitation(x) / self.mask_offset[:, None]
        return np.maximum(masked, self.absolute_threshold[:, None])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _calibrate_norm(self) -> float:
        """Reciprocal peak magnitude of the full-scale calibration sine.

        Feeding a 1019.5 Hz full-scale sine and taking the maximum spectral
        coefficient over 10 frames, per the source of eq (5); the reciprocal is
        used so the calibration tone maps to ``playback_db_spl``.
        """
        n_samples = self.hop * (CALIBRATION_FRAMES + 2) + self.n_fft
        t = np.arange(n_samples) / self.sample_rate
        sine = np.sin(2.0 * np.pi * CALIBRATION_HZ * t)
        frames = self._frames(sine)
        windowed = frames * self.window[None, :]
        spectrum = np.fft.fft(windowed, n=self.n_fft, axis=1) / self.n_fft
        peak = float(np.max(np.abs(spectrum)))
        if peak <= 0.0:
            raise RuntimeError("calibration sine produced a zero spectrum")
        return 1.0 / peak


def tone_signal(
    freq_hz: float,
    amplitude: float,
    n_samples: int,
    sample_rate: int,
    phase: float = 0.0,
) -> np.ndarray:
    """Deterministic sine burst for probe experiments."""
    t = np.arange(n_samples) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t + phase)
