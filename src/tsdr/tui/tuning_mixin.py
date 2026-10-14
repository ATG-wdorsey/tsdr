from __future__ import annotations

import logging
from typing import Literal

from textual import on

from tsdr.core.band_stack import (
    REGISTERS_PER_BAND,
    BandRegister,
    get_band_stack,
    suspended_writeback,
)
from tsdr.core.demod_spec import PreviousTuneState
from tsdr.core.events.events import (
    BandStackChangedEvent,
    TuningStateChangedEvent,
)
from tsdr.core.landmarks import next_target
from tsdr.core.preferences import save_tuning_state
from tsdr.core.sdr.config import LoOffset, TuningMode
from tsdr.core.sdr.device_context import SDRDeviceContext
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.search import TRIGGER_ABOVE_SQUELCH_DB
from tsdr.core.sdr.spectrum_view import adjusted_span
from tsdr.core.sdr.tune_policy import effective_lo_side
from tsdr.core.search_control import (
    jump_targets,
    lock_out_channel,
    search_jump,
    set_search_edge,
    set_search_step,
    set_voice_squelch,
    start_search,
    stop_search,
    unlock,
)
from tsdr.core.tuning import (
    bandwidth_step,
    current_channel_bandwidth,
    current_spec_or_default,
    default_bandwidth,
    next_step,
    resolve_auto_step,
    save_previous_tune_state,
    snap_to_grid,
    tune_state,
)
from tsdr.core.tuning_state import get_tuning_state
from tsdr.core.units import format_hz
from tsdr.radio.registry import DEMODULATORS
from tsdr.tui._mixin_base import MixinBase
from tsdr.tui.messages import DeviceStateChanged, FFTUpdate

logger = logging.getLogger(__name__)


class TuningMixin(MixinBase):
    """Frequency, bandwidth, band-stack, step-ladder, A/B, landmark actions."""

    @on(FFTUpdate)
    def _tuning_on_fft_update(self, message: FFTUpdate) -> None:
        self._latest_fft_by_device[message.event.device_id] = message.event

    @on(DeviceStateChanged)
    def _tuning_on_device_state_changed(self, message: DeviceStateChanged) -> None:
        if not message.event.running:
            self._latest_fft_by_device.pop(message.event.device_id, None)

    def _focused(self) -> SDRDeviceContext | None:
        return get_engine().get_focused_device()

    def _resolve_step(self, device: SDRDeviceContext) -> float:
        step = get_tuning_state().step_for(device.active_mode)
        if step is not None:
            return step
        return resolve_auto_step(device.active_mode, device.config.tuned_frequency)

    def _clamp_to_range(self, device: SDRDeviceContext, freq: float) -> float:
        rng = device.device.capabilities.frequency_range
        if rng is not None:
            lo, hi = rng
            freq = max(lo, min(freq, hi))
        return max(freq, 1.0)  # bands can start at 0 Hz; 0 is invalid

    def _publish_tuning_state_changed(self) -> None:
        ts = get_tuning_state()
        save_tuning_state(ts)
        get_engine().event_bus.publish(TuningStateChangedEvent(state=ts))

    def _apply_freq(self, device: SDRDeviceContext, new_freq: float) -> None:
        new_freq = self._clamp_to_range(device, new_freq)
        if new_freq == device.config.tuned_frequency:
            return
        get_engine().update_device_config(device.device_id, tuned_frequency=new_freq)

    def _tune(self, direction: int, *, coarse: bool = False, fine: bool = False) -> None:
        device = self._focused()
        if device is None:
            return
        step = self._resolve_step(device)
        if coarse:
            step *= 10
        if fine:
            step /= 10
        current = float(device.config.tuned_frequency)
        if fine:
            new_freq = current + direction * step
        else:
            new_freq = snap_to_grid(current, step, direction)
        self._apply_freq(device, new_freq)
        self.show_status(f"Freq: {format_hz(new_freq, interval=step, long_suffix=True)}")

    def _adjust_channel_bandwidth(self, direction: int, *, fine: bool = False) -> None:
        device = self._focused()
        if device is None:
            return
        if device.demod_profile is None:
            self._show_error("No active channel")
            return
        current_bw = int(current_channel_bandwidth(device))
        step = bandwidth_step(current_bw)
        if fine:
            step = max(1, step // 10)
            new_bw = current_bw + direction * step
        else:
            new_bw = int(snap_to_grid(current_bw, step, direction))
        upper = int(device.config.sample_rate / 2)
        new_bw = max(step, min(new_bw, upper))
        if new_bw == current_bw:
            return
        get_engine().update_device_config(device.device_id, channel_bandwidth=new_bw)
        self.show_status(f"Bandwidth: {format_hz(new_bw, decimals=6, long_suffix=True)}")

    def _adjust_spectrum_span(self, direction: int) -> None:
        device = self._focused()
        if device is None:
            return
        new_span = adjusted_span(device.config, device.device.capabilities, direction)
        if new_span == device.config.spectrum_span:
            return
        try:
            get_engine().update_device_config(device.device_id, spectrum_span=new_span)
        except SDRException as e:
            self._show_error(str(e))

    def _toggle_center_tuning(self) -> None:
        device = self._focused()
        if device is None:
            return
        new_mode: TuningMode = "free" if device.config.tuning_mode == "center" else "center"
        if new_mode == "center" and not device.device.capabilities.frequency_controllable:
            self._show_error("Center locked by device")
            return
        get_engine().update_device_config(device.device_id, tuning_mode=new_mode)
        self.show_status(f"Tuning: {new_mode}")

    def _toggle_lo_offset(self, side: LoOffset) -> None:
        """`o`/`O`: put the LO on that side of the dial; the same key again turns it off."""
        device = self._focused()
        if device is None:
            return
        caps = device.device.capabilities
        if not caps.frequency_controllable or caps.provides_spectrum:
            self._show_error("LO offset not available for this device")
            return
        new: LoOffset = "off" if effective_lo_side(device.config.lo_offset, caps) == side else side
        get_engine().update_device_config(device.device_id, lo_offset=new)
        config = device.config
        offset = config.center_frequency - config.tuned_frequency
        if new == "off":
            self.show_status("LO offset: off")
        elif offset:
            where = "below" if offset < 0 else "above"
            size = format_hz(abs(offset), decimals=1, long_suffix=True)
            self.show_status(f"LO offset: {size} {where} the dial")
        elif device.demod_profile is None:
            self.show_status(f"LO offset: {new} the dial while demodulating")
        else:
            self.show_status(f"LO offset: {new} the dial (no room at this sample rate)")

    def _toggle_search(self) -> None:
        """`/`: start SEARCH with the last settings, or stop it."""
        device = self._focused()
        if device is None:
            return
        try:
            if stop_search(device.device_id):
                self.show_status("SEARCH off")
                return
            search = start_search(device.device_id)
        except SDRException as e:
            self._show_error(str(e))
            return
        # A tuning step coarser than the channel would blind SEARCH between its grid
        # points; it searched on the mode's automatic step, so the tuner shows that.
        ts = get_tuning_state()
        step = ts.step_for(device.active_mode)
        coarse = ""
        if step is not None and search.step_hz < step:
            coarse = (
                f"; step {format_hz(search.step_hz, decimals=1, long_suffix=True)} (auto: the "
                f"{format_hz(step, decimals=1, long_suffix=True)} step would miss stations)"
            )
            ts.set_step(device.active_mode, None)
            self._publish_tuning_state_changed()
        low = format_hz(search.low_hz, decimals=3, long_suffix=True)
        high = format_hz(search.high_hz, decimals=3, long_suffix=True)
        trigger = (
            f"squelch + {TRIGGER_ABOVE_SQUELCH_DB:g} dB"
            if search.trigger_db is None
            else f"{search.trigger_db:.0f} dBFS"
        )
        self.show_status(f"SEARCH {low} to {high}, trigger {trigger}{coarse}")

    def _set_search_edge(self, edge: Literal["low", "high"]) -> None:
        """`{`/`}`: put SEARCH's lower/upper edge at the dial."""
        device = self._focused()
        if device is None:
            return
        try:
            frequency = set_search_edge(device.device_id, edge)
        except SDRException as e:
            self._show_error(str(e))
            return
        where = "lower" if edge == "low" else "upper"
        self.show_status(
            f"SEARCH {where} edge: {format_hz(frequency, decimals=4, long_suffix=True)}"
        )

    def _lock_out_channel(self, *, permanent: bool) -> None:
        """`x`/`X`: skip the dial's channel, for this session or as a memory."""
        device = self._focused()
        if device is None:
            return
        try:
            frequency = lock_out_channel(device.device_id, permanent=permanent)
        except SDRException as e:
            self._show_error(str(e))
            return
        where = "saved as a memory" if permanent else "this session"
        self.show_status(
            f"Locked out {format_hz(frequency, decimals=4, long_suffix=True)} ({where})"
        )

    def _unlock(self) -> None:
        """`ctrl+x`: unlock the dial's channel, or else every session lockout."""
        device = self._focused()
        if device is None:
            return
        result = unlock(device.device_id)
        if result.frequency_hz is not None:
            deleted = " (lockout memory deleted)" if result.saved else ""
            where = format_hz(result.frequency_hz, decimals=4, long_suffix=True)
            self.show_status(f"Unlocked {where}{deleted}")
        elif result.session:
            plural = "s" if result.session != 1 else ""
            self.show_status(f"Dropped {result.session} session lockout{plural}")
        else:
            self.show_status("No session lockouts to drop")

    def _toggle_vsq(self) -> None:
        """`v`: VSQ, SEARCH skipping transmissions without speech."""
        device = self._focused()
        if device is None:
            return
        try:
            on = set_voice_squelch(device.device_id)
        except SDRException as e:
            self._show_error(str(e))
            return
        self.show_status("VSQ on: SEARCH skips transmissions without speech" if on else "VSQ off")

    def _center_view_on_dial(self) -> None:
        device = self._focused()
        if device is None:
            return
        if device.config.spectrum_center is None:
            self.show_status("View already on dial")
            return
        get_engine().update_device_config(device.device_id, spectrum_center=None)
        self.show_status("View: dial")

    def _cycle_step(self, forward: bool) -> None:
        """Step through the ladder for the focused device's mode; each mode keeps its own."""
        device = self._focused()
        if device is None:
            self._show_error("No device")
            return
        mode = device.active_mode
        ts = get_tuning_state()
        chosen = next_step(mode, ts.step_for(mode), forward)
        ts.set_step(mode, chosen)
        self._publish_tuning_state_changed()

        searched = ""
        if device.config.search is not None:
            step = chosen or resolve_auto_step(mode, device.config.tuned_frequency)
            used = set_search_step(device.device_id, step)
            if used != step:
                searched = f" (SEARCH: {format_hz(used, decimals=1, long_suffix=True)})"
        if chosen is None:
            resolved = resolve_auto_step(mode, device.config.tuned_frequency)
            self.show_status(
                f"Step: auto ({format_hz(resolved, decimals=1, long_suffix=True)}){searched}"
            )
        else:
            self.show_status(f"Step: {format_hz(chosen, decimals=1, long_suffix=True)}{searched}")

    def _jump_target(self, direction: int) -> None:
        device = self._focused()
        if device is None:
            return
        targets, limits = jump_targets(device, self._latest_fft_by_device.get(device.device_id))
        target = next_target(direction, float(device.config.tuned_frequency), targets, limits)
        searching = device.config.search is not None
        if target is None:
            where = (
                "signal further in the search range"
                if searching
                else "signal or memory further in the band"
            )
            self.show_status(f"No {where}")
            return
        before = tune_state(device)
        if searching:
            jumped = search_jump(device.device_id, target)
        elif device.config.tuning_mode == "free":
            # In free tuning `[` / `]` never move the LO.
            jumped = get_engine().update_device_config(
                device.device_id, keep_lo=True, tuned_frequency=target
            )
        else:
            self._apply_freq(device, target)
            jumped = True
        if not jumped:
            self.show_status("That target would move the LO; the dial stays")
            return
        save_previous_tune_state(device, before)
        self.show_status(f"Jump: {format_hz(target, interval=1.0, long_suffix=True)}")

    def _swap_ab(self) -> None:
        device = self._focused()
        if device is None:
            return
        ts = get_tuning_state()
        if ts.previous is None:
            self.show_status("No previous tune state")
            return
        cur_spec = current_spec_or_default(device)
        captured = PreviousTuneState(
            frequency_hz=float(device.config.tuned_frequency),
            bandwidth_hz=current_channel_bandwidth(device),
            spec=cur_spec,
        )
        prev = ts.previous
        if captured == prev:
            return
        rng = device.device.capabilities.frequency_range
        if rng is not None and not (rng[0] <= prev.frequency_hz <= rng[1]):
            self._show_error(
                f"A/B: {format_hz(prev.frequency_hz, interval=1.0, long_suffix=True)} "
                f"out of device range"
            )
            return
        ts.previous = captured

        engine = get_engine()
        if prev.spec.mode and prev.spec.mode != device.active_mode and prev.spec.mode != "RAW":
            try:
                engine.set_audio_demod(device.device_id, prev.spec)
            except SDRException as e:
                self._show_error(str(e))
                return
        engine.update_device_config(
            device.device_id,
            tuned_frequency=prev.frequency_hz,
            channel_bandwidth=int(prev.bandwidth_hz),
        )
        self._publish_tuning_state_changed()
        self.show_status(
            f"A/B: {format_hz(prev.frequency_hz, interval=1.0, long_suffix=True)} {prev.spec.mode}"
        )

    def _recall_band(self, key: int) -> None:
        device = self._focused()
        if device is None:
            return
        store = get_band_stack()
        stack = store.get_by_key(key)
        if stack is None:
            self._show_error(f"No band on key {key}")
            return

        cur_freq = float(device.config.tuned_frequency)
        ts = get_tuning_state()
        on_this_band = stack.band.start <= cur_freq <= stack.band.end and ts.current_band_key == key
        new_idx = (
            (stack.current_idx + 1) % REGISTERS_PER_BAND if on_this_band else stack.current_idx
        )

        reg = store.get_register(key, new_idx)
        if reg is None:
            seed_mode = device.active_mode if device.active_mode != "RAW" else "AM"
            seed_bw = int(current_channel_bandwidth(device)) or default_bandwidth(seed_mode)
            seed_spec = current_spec_or_default(device, override_mode=seed_mode)
            reg = BandRegister(
                slot=new_idx,
                frequency=(stack.band.start + stack.band.end) // 2,
                audio_spec=seed_spec,
                bandwidth=int(seed_bw),
            )
            store.set_register(key, new_idx, reg)

        rng = device.device.capabilities.frequency_range
        if rng is not None and not (rng[0] <= reg.frequency <= rng[1]):
            self._show_error(
                f"Band {stack.band.name}: "
                f"{format_hz(reg.frequency, interval=1.0, long_suffix=True)} "
                f"out of device range"
            )
            return

        save_previous_tune_state(device)

        engine = get_engine()
        with suspended_writeback():
            if reg.audio_spec.mode != device.active_mode and reg.audio_spec.mode in DEMODULATORS:
                try:
                    engine.set_audio_demod(device.device_id, reg.audio_spec)
                except SDRException as e:
                    self._show_error(str(e))
                    return
            try:
                engine.update_device_config(
                    device.device_id,
                    tuned_frequency=float(reg.frequency),
                    channel_bandwidth=int(reg.bandwidth),
                )
            except SDRException as e:
                self._show_error(str(e))
                return
        store.set_current_idx(key, new_idx)
        ts.current_band_key = key
        self._publish_tuning_state_changed()
        get_engine().event_bus.publish(BandStackChangedEvent(band_stack=store))
        self.show_status(f"BAND {stack.band.name} REG {new_idx + 1}/{REGISTERS_PER_BAND}")

    def notify_demod_changed(self) -> None:
        """Called when the mode changes via a demod chord: the tuner shows the new mode's step."""
        self._publish_tuning_state_changed()
