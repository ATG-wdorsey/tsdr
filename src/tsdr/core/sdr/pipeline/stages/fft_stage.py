import numpy as np

from tsdr.core.sdr.buffers import CircularBuffer
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.processing import (
    apply_dc_offset_correction,
    apply_iq_imbalance_correction,
    compute_fft,
    fft_window,
)
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import span, traced
from tsdr.radio.dsp._kernels import add_power_c64, iq_sums_c64, window_corrected_into
from tsdr.radio.dsp.fft import FFTPlan

# FFT frames SEARCH averages per batch (spread over it) while it runs.
SEARCH_FRAMES = 8


class FFTStage:
    """Stage that computes FFT spectrum and frequency axis.

    Accumulates IQ samples in a circular buffer, computes FFT when
    enough samples are available.
    """

    def __init__(self, config: DeviceConfig):
        self.fft_size = config.fft_size
        self.window_type = config.fft_window

        # Create circular buffer (2x size for overlap)
        self.circular_buffer = CircularBuffer(size=self.fft_size * 2, dtype=np.complex64)

        # FFTW plan (reused across calls)
        self._fft_plan = FFTPlan(self.fft_size)

        # Cached window, normalization factor, and frequency axis
        self._cached_window: np.ndarray | None = None
        self._cached_inv_norm_sq: float = 0.0
        self._cached_fftfreq: np.ndarray | None = None
        self._cached_window_size: int = 0
        self._cached_window_type: str = ""
        self._cached_sample_rate: float = 0.0

        # Averaged spectrum for EMA smoothing
        self._averaged_spectrum: np.ndarray | None = None
        self._spectrum_ema_scratch: np.ndarray | None = None

    def _get_or_create_window(self) -> tuple[np.ndarray, float]:
        """Get cached window and normalization factor, creating if needed.

        Returns:
            (window, inv_norm_sq) where inv_norm_sq = 1/(coherent_gain * N)^2
        """
        if (
            self._cached_window is None
            or self._cached_window_size != self.fft_size
            or self._cached_window_type != self.window_type
        ):
            self._cached_window = fft_window(self.window_type, self.fft_size)

            coherent_gain = float(np.sum(self._cached_window)) / self.fft_size
            self._cached_inv_norm_sq = 1.0 / (coherent_gain * self.fft_size) ** 2
            self._cached_window_size = self.fft_size
            self._cached_window_type = self.window_type
            self._cached_fftfreq = None  # invalidate freq axis too

        return self._cached_window, self._cached_inv_norm_sq

    def _get_or_create_fftfreq(self, sample_rate: float) -> np.ndarray:
        """Get cached fftshift'd frequency bins, creating if needed."""
        if (
            self._cached_fftfreq is None
            or self._cached_window_size != self.fft_size
            or self._cached_sample_rate != sample_rate
        ):
            self._cached_fftfreq = np.fft.fftshift(
                np.fft.fftfreq(self.fft_size, 1.0 / sample_rate)
            ).astype(np.float64)
            self._cached_sample_rate = sample_rate
        return self._cached_fftfreq

    def _search_spectrum(
        self, iq: np.ndarray, config: SDRConfig | None, latest: np.ndarray
    ) -> np.ndarray:
        """SEARCH's spectrum: the power of SEARCH_FRAMES frames spread over the batch,
        averaged (one frame of noise reads about 1 dB either side of its mean in a 12.5 kHz
        channel at 32768 points). The newest frame is the display's. The others get the
        display's corrections, fused with the window into the FFT's input in one pass."""
        n = self.fft_size
        if len(iq) < 2 * n:
            return latest
        window, inv_norm_sq = self._get_or_create_window()
        frames = min(SEARCH_FRAMES, len(iq) // n)
        starts = np.linspace(0, len(iq) - n, frames).astype(np.intp)[:-1]
        dc = config is not None and config.dc_offset_correction
        balance = config is not None and config.iq_imbalance_correction
        total = np.zeros(n, dtype=np.float32)
        for start in starts:
            segment = iq[start : start + n]
            dc_i = dc_q = 0.0
            q_scale = 1.0
            if dc or balance:
                sr, si, srr, sii = iq_sums_c64(segment)
                if dc:
                    dc_i, dc_q = sr / n, si / n
                var_i = srr / n - (sr / n) ** 2
                var_q = sii / n - (si / n) ** 2
                if balance and var_i > 0 and var_q > 0:
                    q_scale = float(np.sqrt(var_i / var_q))
            window_corrected_into(segment, window, dc_i, dc_q, q_scale, self._fft_plan.input)
            add_power_c64(self._fft_plan.run(), total)
        total = np.fft.fftshift(total)
        total *= np.float32(inv_norm_sq)
        # exp and log, not power and log10: numpy's float32 power and log10 take 10-15x longer.
        total += np.exp(latest * np.float32(np.log(10.0) / 10.0))
        total *= np.float32(1.0 / frames)
        total += np.float32(1e-20)
        np.log(total, out=total)
        total *= np.float32(10.0 / np.log(10.0))
        return total

    @traced("fft")
    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        """Compute FFT from IQ samples.

        Args:
            data: Pipeline data with iq_samples
            context: Pipeline context

        Returns:
            SamplesBatch with spectrum and frequencies populated, or None if not enough samples
        """
        if data.iq_samples is None:
            return None

        with span("buffer_append"):
            self.circular_buffer.append(data.iq_samples)

        if self.circular_buffer.size < self.fft_size:
            return None

        with span("get_window"):
            iq_window = self.circular_buffer.get_window(self.fft_size)

        config = context.config
        if config is not None:
            if config.dc_offset_correction:
                with span("dc_offset_correction"):
                    iq_window = apply_dc_offset_correction(iq_window)

            if config.iq_imbalance_correction:
                with span("iq_imbalance_corr"):
                    iq_window = apply_iq_imbalance_correction(iq_window)

        with span("compute_fft"):
            window, inv_norm_sq = self._get_or_create_window()
            spectrum = compute_fft(iq_window, window, inv_norm_sq, self._fft_plan)
        instant_spectrum = spectrum
        device = context.device_context
        if SEARCH_FRAMES > 1 and device is not None and device.config.search is not None:
            with span("search_frames"):
                instant_spectrum = self._search_spectrum(data.iq_samples, config, spectrum)

        # Apply exponential moving average for spectrum smoothing
        # alpha = 1/N where N is spectrum_averaging: N=1 means no averaging, higher = more smoothing
        if config is not None and config.spectrum_averaging > 1:
            alpha = np.float32(1.0 / config.spectrum_averaging)
            avg = self._averaged_spectrum
            scratch = self._spectrum_ema_scratch
            if (
                avg is None
                or scratch is None
                or avg.shape != spectrum.shape
                or avg.dtype != spectrum.dtype
            ):
                avg = np.empty_like(spectrum)
                np.copyto(avg, spectrum)
                self._averaged_spectrum = avg
                self._spectrum_ema_scratch = np.empty_like(spectrum)
            else:
                # avg += alpha * (spectrum - avg), all in place via scratch
                np.subtract(spectrum, avg, out=scratch)
                scratch *= alpha
                avg += scratch
            spectrum = avg

        freq_bins = self._get_or_create_fftfreq(data.sample_rate)
        frequencies: np.ndarray = (data.center_frequency + freq_bins).astype(np.float32)  # type: ignore[assignment]

        return data.with_changes(
            spectrum=spectrum,
            instant_spectrum=instant_spectrum,
            frequencies=frequencies,
            stage_name="fft",
        )

    def on_config_change(self, config) -> None:
        if not isinstance(config, DeviceConfig):
            return
        if config.fft_size != self.fft_size:
            self.fft_size = config.fft_size
            self.circular_buffer = CircularBuffer(size=self.fft_size * 2, dtype=np.complex64)
            self._fft_plan = FFTPlan(self.fft_size)
            # Invalidate all caches (length changes)
            self._cached_window = None
            self._cached_fftfreq = None
            self._averaged_spectrum = None
            self._spectrum_ema_scratch = None

        if config.fft_window != self.window_type:
            self.window_type = config.fft_window
            self._cached_window = None

    def reset(self) -> None:
        """Reset stage state.

        Clears circular buffer and averaged spectrum to remove stale data.
        Window cache is preserved as it's configuration-dependent.
        """
        self.circular_buffer.clear()
        self._averaged_spectrum = None
