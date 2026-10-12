from argparse import Namespace

from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.tui.commands._format import device_id, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


class SDRStartCommand(Command):
    @property
    def description(self) -> str:
        return "Start an SDR device"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument("device_id", nargs="?")

    def run(self, args: Namespace) -> str:
        engine = get_engine()
        did = args.device_id or get_focused_device_id()
        shown = engine.get_focused_device()
        engine.start_device(did)
        # The display and the keys follow the focused device: don't leave them
        # on a stopped one. A device already running keeps the focus.
        if shown is None or (shown.device_id != did and shown.state != DeviceState.RUNNING):
            engine.set_focused_device(did)
            return success(f"Started {device_id(did)} (now in view)")
        return success(f"Started {device_id(did)}")

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        return device_id_completions(prefix)
