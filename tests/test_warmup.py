"""The kernel warm-up runs through: a kernel whose call it makes no longer fits (an argument
renamed, a type changed) would end it before it notes it has run, and tsdr would start it
again at every start. And it compiles what tsdr then calls: numba compiles a kernel apart for
every signature, a read-only array included (a device's bytes give one)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from numba.core.registry import CPUDispatcher

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.stages.event_emitter_stage import _iq_metrics_c64
from tsdr.core.sdr.pipeline.stages.fft_stage import FFTStage
from tsdr.core.sdr.pipeline.stages.frequency_shift_stage import FrequencyShiftStage
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.radio.registry import DEMODULATOR_CLASSES, make_demodulator
from tsdr.warmup import warm_up


def _compiled() -> dict[str, int]:
    return {
        f"{name}.{attr}": len(obj.signatures)
        for name, module in list(sys.modules.items())
        if name.startswith("tsdr.")
        for attr, obj in vars(module).items()
        if isinstance(obj, CPUDispatcher)
    }


def test_the_warm_up_compiles_what_a_device_s_samples_meet() -> None:
    warm_up()
    before = _compiled()
    rate = 1_000_000.0
    context = SimpleNamespace(
        device_context=SimpleNamespace(config=SimpleNamespace(tuned_frequency=1000.0))
    )
    for sample_format in SampleFormat:
        raw = bytes(sample_format.bytes_per_sample * 4096)
        iq = SamplesBatch(raw_samples=raw, sample_format=sample_format).to_iq_array()
        _iq_metrics_c64(np.ascontiguousarray(iq, dtype=np.complex64))
        FFTStage(DeviceConfig(fft_size=1024))._search_spectrum(
            iq, SDRConfig(), np.zeros(1024, dtype=np.float32)
        )
        batch = SamplesBatch(iq_samples=iq, center_frequency=0.0, sample_rate=rate)
        FrequencyShiftStage().process(batch, context)  # type: ignore[arg-type]
        for mode, cls in DEMODULATOR_CLASSES.items():
            if cls.FIXED_SAMPLE_RATE is None:  # tuned to the capture's center: iq as it came
                make_demodulator(DemodSpec(mode=mode), rate).demodulate(iq, 1.7e9)
    after = _compiled()
    assert {name for name in after if after[name] != before.get(name, 0)} == set()
