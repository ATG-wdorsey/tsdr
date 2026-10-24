"""P25 keeps the capture on the site: free tuning when its decoder comes into service, whichever path
brings it: chosen, restored with a saved configuration, added as a pipeline, or started (review 52, R52-3: a saved P25
configuration started in center tuning, the capture following the dial off the site)."""

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, StageType
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams

STAGES = (
    StageType.FREQUENCY_SHIFT,
    StageType.DEMODULATOR,
    StageType.DENOISER,
    StageType.EVENT_EMITTER,
)


def _p25() -> PipelineConfig:
    return PipelineConfig(stages=STAGES, audio_spec=DemodSpec(mode="P25"))


@pytest.fixture
def engine():
    engine = SDREngine()
    yield engine
    for context in engine.devices.values():
        context.state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _add(engine: SDREngine, pipelines: dict) -> None:
    engine.add_device(
        "radio",
        "mock",
        MockParams(),
        DeviceConfig(
            sample_rate=8e6,
            tuned_frequency=858e6,
            center_frequency=858e6,
            tuning_mode="center",
            pipelines=pipelines,
        ),
    )


def test_a_restored_p25_configuration_starts_in_free_tuning(engine: SDREngine) -> None:
    _add(engine, {"audio": _p25()})
    assert engine.devices["radio"].config.tuning_mode == "free"


def test_starting_a_device_with_p25_selects_free_tuning(engine: SDREngine) -> None:
    _add(engine, {})
    context = engine.devices["radio"]
    # an explicit choice made with the change itself stays (configuration)...
    engine.update_device_config("radio", pipelines={"audio": _p25()}, tuning_mode="center")
    assert context.config.tuning_mode == "center"
    # ...until P25 starts: the workers aren't run, only the engine's start policy
    context.start = lambda worker_runner: setattr(context, "state", DeviceState.RUNNING)
    engine.start_audio_output = lambda *args: None
    engine.start_device("radio")
    assert context.config.tuning_mode == "free"


def test_adding_a_p25_pipeline_selects_free_tuning(engine: SDREngine) -> None:
    _add(engine, {})
    engine.add_pipeline("radio", "audio", _p25())
    assert engine.devices["radio"].config.tuning_mode == "free"


def test_a_tuning_mode_chosen_while_p25_runs_stays(engine: SDREngine) -> None:
    # only P25 coming into service selects it: a squelch change or a recording keeps a choice made since (review 53,
    # R53-4)
    _add(engine, {"audio": _p25()})
    context = engine.devices["radio"]
    engine.update_device_config("radio", tuning_mode="center")  # `config --tuning center`
    engine.update_squelch("radio", "audio", enabled=True, threshold_db=-40.0)
    engine.add_pipeline(
        "radio", "record", PipelineConfig(stages=(StageType.RECORD,), record_path="/dev/null")
    )
    engine.remove_pipeline("radio", "record")
    assert context.config.tuning_mode == "center"
