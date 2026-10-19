from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.stages.demodulator_stage import DemodulatorStage
from tsdr.radio.decoders.p25.decoder import P25Decoder
from tsdr.radio.decoders.p25.talkgroups import TalkgroupTable
from tsdr.tui.commands._format import success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

_ON_OFF = {"on": True, "off": False}


def _table(device_id: str) -> TalkgroupTable:
    """The talkgroup table of the system the device's P25 decoder hears, or raise SDRException."""
    context = get_engine().get_device(device_id)
    for pipeline in context.pipelines.values():
        for stage in pipeline.stages:
            if isinstance(stage, DemodulatorStage) and isinstance(stage.demodulator, P25Decoder):
                table = stage.demodulator.talkgroup_table
                if table is None:
                    raise SDRException("No P25 system heard yet (its control channel names it)")
                return table
    raise SDRException(f"No P25 decoder active on device '{device_id}'")


class P25Command(Command):
    @property
    def description(self) -> str:
        return "P25 trunking: the talkgroups followed (also the P25 panel's keys, Tab)"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="subcommand")
        follow = sub.add_parser("follow", help="follow <TG>... on|off, or follow all|none")
        follow.add_argument("targets", nargs="+")
        new = sub.add_parser("new", help="whether newly heard talkgroups are followed")
        new.add_argument("state", choices=sorted(_ON_OFF))
        rescan = sub.add_parser("rescan", help="learn a talkgroup's encryption again")
        rescan.add_argument("groups", nargs="+", type=int)
        sub.add_parser("talkgroups", help="list the talkgroups heard")
        for p in (follow, new, rescan, sub.choices["talkgroups"]):
            p.add_argument("--device", dest="device_id")

    def run(self, args: Namespace) -> str:
        if not args.subcommand:
            return self.help_text()
        table = _table(getattr(args, "device_id", None) or get_focused_device_id())
        if args.subcommand == "follow":
            return self._follow(table, args.targets)
        if args.subcommand == "new":
            table.set_follow_new(_ON_OFF[args.state])
            return success(
                f"Newly heard talkgroups: {'followed' if table.follow_new else 'not followed'}"
            )
        if args.subcommand == "rescan":
            for group in args.groups:
                table.rescan(group)
            return success(
                f"Encryption learned again at the next encrypted call: {', '.join(map(str, args.groups))}"
            )
        if args.subcommand == "talkgroups":
            rows = table.snapshot()
            if not rows:
                return "No talkgroups heard yet"
            return "\n".join(
                f"  TG {t.group:<6} {'[cyan]followed[/]' if t.follow else '[dim]not followed[/]'}"
                + (
                    f"  ALGID 0x{t.algid:02X} key 0x{t.kid:04X}"
                    if t.algid is not None and t.kid is not None
                    else ""
                )
                for t in rows
            )
        raise SDRException(f"Unknown subcommand: {args.subcommand}")

    @staticmethod
    def _follow(table: TalkgroupTable, targets: list[str]) -> str:
        if targets in (["all"], ["none"]):
            table.set_all(targets == ["all"])
            return success(
                "Following every talkgroup heard, and newly heard ones"
                if targets == ["all"]
                else "Following none, nor newly heard ones"
            )
        *groups, state = targets
        if state not in _ON_OFF or not groups or not all(g.isdigit() for g in groups):
            raise SDRException("Usage: p25 follow <TG>... on|off, or p25 follow all|none")
        for group in groups:
            table.set_follow(int(group), _ON_OFF[state])
        return success(
            f"TG {', '.join(groups)}: {'followed' if _ON_OFF[state] else 'not followed'}"
        )

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
        if subcommand == "follow":
            return [Completion(w) for w in ("all", "none", "on", "off") if w.startswith(prefix)]
        return []
