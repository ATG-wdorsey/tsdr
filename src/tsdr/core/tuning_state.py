from __future__ import annotations

from dataclasses import dataclass, field

from tsdr.core.demod_spec import PreviousTuneState


@dataclass
class TuningState:
    # The step the user last chose in each mode (NFM, USB, RAW, ...); a mode not here
    # tunes by its own automatic step until the user picks one.
    steps: dict[str, float] = field(default_factory=dict)
    previous: PreviousTuneState | None = None
    current_band_key: int | None = None  # 1..9 — None when not on a band

    def step_for(self, mode: str) -> float | None:
        """The user's step in `mode`; None = auto (the mode's own step)."""
        return self.steps.get(mode)

    def set_step(self, mode: str, step: float | None) -> None:
        if step is None:
            self.steps.pop(mode, None)
        else:
            self.steps[mode] = float(step)


_state: TuningState | None = None


def get_tuning_state() -> TuningState:
    if _state is None:
        raise RuntimeError("TuningState not initialized")
    return _state


def init_tuning_state() -> TuningState:
    global _state
    _state = TuningState()
    return _state
