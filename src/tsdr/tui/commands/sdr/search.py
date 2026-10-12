from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.search import LOCKOUT_TAG, TRIGGER_ABOVE_SQUELCH_DB, SearchConfig
from tsdr.core.search_control import (
    clear_session_lockouts,
    lock_out_channel,
    search_settings,
    start_search,
    stop_search,
)
from tsdr.core.units import parse_hz
from tsdr.tui.commands._format import fields, freq_mhz, state, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id


def _dbfs(value: float) -> str:
    return f"[yellow]{value:.1f} dBFS[/]"


def describe(search: SearchConfig, squelch_db: float) -> str:
    dwell = "unlimited" if search.max_dwell_s is None else f"{search.max_dwell_s:g} s"
    lockouts = len(search.lockouts) + len(search.memory_lockouts)
    return fields(
        {
            "range": f"{freq_mhz(search.low_hz, precision=4)}-{freq_mhz(search.high_hz, precision=4)}",
            "step": f"[yellow]{search.step_hz / 1e3:g} kHz[/]",
            "trigger": (
                f"squelch + {TRIGGER_ABOVE_SQUELCH_DB:g} dB"
                if search.trigger_db is None
                else _dbfs(search.trigger_db)
            ),
            "hold": f"{_dbfs(squelch_db)} for {search.hold_s:g} s",
            "dwell": dwell,
            "lockouts": str(lockouts),
        }
    )


class SDRSearchCommand(Command):
    @property
    def description(self) -> str:
        return "SEARCH the captured band and stop on active channels"

    def configure(self, parser: CommandParser) -> None:
        parser.add_argument(
            "action",
            nargs="?",
            choices=["on", "off", "lockout", "unlock"],
            help=(
                "on: start (or restart with new settings); off: stop; lockout: skip the "
                "current channel; unlock: drop this session's lockouts"
            ),
        )
        parser.add_argument(
            "--low", help="Lower edge with SI suffix (default: the captured band's)"
        )
        parser.add_argument(
            "--high", help="Upper edge with SI suffix (default: the captured band's)"
        )
        parser.add_argument(
            "--full",
            action="store_true",
            help="Search the whole captured band: forget the edges set with --low/--high or { }",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="Trigger in dBFS within the channel, as the squelch measures; must be above "
            "the squelch, which holds the channel (default: 6 dB above the squelch, "
            "following it as it changes)",
        )
        parser.add_argument(
            "--hold",
            type=float,
            help="Seconds a channel stays held after it drops below the squelch (default 2)",
        )
        parser.add_argument(
            "--dwell",
            type=float,
            help="Most seconds on one active channel; 0 = unlimited (the default)",
        )
        parser.add_argument(
            "--permanent",
            action="store_true",
            help=f"With lockout: save a memory tagged '{LOCKOUT_TAG}' instead of this session only",
        )
        parser.add_argument("--device", dest="device_id")

    def run(self, args: Namespace) -> str:
        did = args.device_id or get_focused_device_id()
        settings_given = args.full or any(
            v is not None for v in (args.low, args.high, args.threshold, args.hold, args.dwell)
        )
        if args.action == "off":
            return success("SEARCH stopped") if stop_search(did) else "SEARCH is not running"
        if args.action == "lockout":
            frequency = lock_out_channel(did, permanent=args.permanent)
            scope = f"memory tagged '{LOCKOUT_TAG}'" if args.permanent else "this session"
            return success(f"Locked out {freq_mhz(frequency, precision=4)} ({scope})")
        if args.action == "unlock":
            count = clear_session_lockouts(did)
            return success(f"Dropped {count} session lockout{'s' if count != 1 else ''}")
        if args.action == "on" or settings_given:
            search = start_search(
                did,
                low_hz=float(parse_hz(args.low)) if args.low else None,
                high_hz=float(parse_hz(args.high)) if args.high else None,
                trigger_db=args.threshold,
                hold_s=args.hold,
                max_dwell_s=args.dwell,
                full_band=args.full,
            )
            squelch_db = get_engine().get_device(did).config.pipelines["audio"].squelch_threshold_db
            return success(f"SEARCH {state('on')} ({describe(search, squelch_db)})")

        config = get_engine().get_device(did).config
        if config.search is None:
            lockouts = len(search_settings(did).lockouts)
            return f"SEARCH {state('off')} ({lockouts} session lockouts)"
        squelch_db = config.pipelines["audio"].squelch_threshold_db
        return f"SEARCH {state('on')} ({describe(config.search, squelch_db)})"

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag == "--device":
            return device_id_completions(prefix)
        return []
