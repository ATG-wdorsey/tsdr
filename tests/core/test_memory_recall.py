"""A recalled memory applies everything it stores, as a radio's memory does, also in the
mode already in use: a DMR repeater's timeslot was ignored when recalled while in DMR."""

from __future__ import annotations

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.memories import Memory, recall_memory
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.tuning_state import init_tuning_state
from tsdr.devices import MockParams

DIAL = 151.3e6


def test_a_recall_in_the_mode_in_use_applies_its_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SDREngine()
    engine.add_device(
        "m", "mock", MockParams(), DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL)
    )
    engine.set_audio_demod("m", DemodSpec(mode="DMR"))
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    init_tuning_state()
    memory = Memory(
        id="m1",
        frequency=int(DIAL) + 25_000,
        name="repeater TS2",
        audio_spec=DemodSpec(mode="DMR", dmr_slot="2"),
        bandwidth=12_500,
    )

    recall_memory(memory, "m")

    audio = engine.devices["m"].config.pipelines["audio"]
    assert audio.audio_spec is not None and audio.audio_spec.dmr_slot == "2"
