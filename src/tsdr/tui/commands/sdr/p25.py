import math
from argparse import Namespace

from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.stages.demodulator_stage import DemodulatorStage
from tsdr.radio.decoders.p25.decoder import P25Decoder, rate_text
from tsdr.radio.decoders.p25.systems import SystemState
from tsdr.tui.commands._format import success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

_ON_OFF = {"on": True, "off": False}


def _decoder(device_id: str) -> P25Decoder:
    """The device's P25 decoder, or raise SDRException."""
    context = get_engine().get_device(device_id)
    for pipeline in context.pipelines.values():
        for stage in pipeline.stages:
            if isinstance(stage, DemodulatorStage) and isinstance(stage.demodulator, P25Decoder):
                return stage.demodulator
    raise SDRException(f"No P25 decoder active on device '{device_id}'")


def _table(device_id: str) -> SystemState:
    """The system the device's P25 decoder hears, or raise SDRException."""
    table = _decoder(device_id).system_state
    if table is None:
        raise SDRException("No P25 system heard yet (its control channel names it)")
    return table


_NO_SITE = "No P25 site heard yet (its control channel and grants give its channels)"


def _movable(device_id: str) -> float:
    """The device's sample rate, or raise SDRException when its capture can't be moved."""
    context = get_engine().get_device(device_id)
    caps = context.device.capabilities
    if not caps.frequency_controllable or caps.provides_spectrum:
        raise SDRException("This device's capture can't be moved")
    return context.config.sample_rate


def center_on_site(device_id: str) -> str:
    """`c` while P25 runs (the owner, 2026-10-20): the capture centered on the site's channels heard at its rate as it
    is, the control channel kept inside; `p25 fit` also raises the rate. The status line's text, or raise
    SDRException."""
    decoder = _decoder(device_id)
    rate = _movable(device_id)
    placed = decoder.center_for(rate)
    if placed is None:
        raise SDRException(_NO_SITE)
    low, high, center, need = placed
    get_engine().update_device_config(device_id, center_frequency=center, tuning_mode="free")
    site = f"{low / 1e6:.4f}-{high / 1e6:.4f}"
    if decoder.holds(
        low, high, center, rate
    ):  # every channel heard inside, whether the LO could be kept clear or not
        return f"P25: capture centered on the site ({site}): {center / 1e6:.4f} MHz"
    return (
        f"P25: capture at {center / 1e6:.4f} MHz, the control channel inside; the site ({site}) needs "
        f"{rate_text(need)} MS/s, the capture is {rate / 1e6:.1f}: channels at its edge stay outside (p25 fit)"
    )


def _fit(device_id: str) -> str:
    """The capture center and rate that hold the site's channels heard, the dial staying on its control channel: the
    rate raised only when the current one can't. When no rate the device has can, the control channel is kept inside,
    as `c` keeps it."""
    decoder = _decoder(device_id)
    fit = decoder.fit()
    if fit is None:
        raise SDRException(_NO_SITE)
    need = fit[3]
    rate = _movable(device_id)
    engine = get_engine()
    caps = engine.get_device(device_id).device.capabilities
    if rate < need:
        if caps.sample_rates:
            rates = sorted(caps.sample_rates)
            rate = next((r for r in rates if r >= need), rates[-1])
        else:
            rate = math.ceil(need / 500_000) * 500_000.0
    placed = decoder.center_for(rate)
    if placed is None:
        raise SDRException(_NO_SITE)
    low, high, center, need = placed
    engine.update_device_config(
        device_id, sample_rate=rate, center_frequency=center, tuning_mode="free"
    )
    held = "" if rate >= need else f" (it needs {rate_text(need)} MS/s: the edges stay out)"
    return success(
        f"Capture {center / 1e6:.4f} MHz, {rate / 1e6:.1f} MS/s: the site's channels heard, "
        f"{low / 1e6:.4f}-{high / 1e6:.4f} MHz{held}"
    )


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
        rescan = sub.add_parser(
            "rescan", help="look again at a talkgroup's next encrypted call (its keys kept)"
        )
        rescan.add_argument("groups", nargs="+", type=int)
        rescan.add_argument(
            "--forget",
            action="store_true",
            help="instead forget what was learned of its encryption (its keys, a flag known wrong)",
        )
        sub.add_parser("talkgroups", help="list the talkgroups heard")
        sub.add_parser(
            "fit", help="set the capture's center and rate to hold the site's channels heard"
        )
        for p in (follow, new, rescan, sub.choices["talkgroups"], sub.choices["fit"]):
            p.add_argument("--device", dest="device_id")

    def run(self, args: Namespace) -> str:
        if not args.subcommand:
            return self.help_text()
        device_id = getattr(args, "device_id", None) or get_focused_device_id()
        if args.subcommand == "fit":
            return _fit(device_id)
        table = _table(device_id)
        if args.subcommand == "follow":
            return self._follow(table, args.targets)
        if args.subcommand == "new":
            table.set_follow_new(_ON_OFF[args.state])
            return success(
                f"Newly heard talkgroups: {'followed' if table.follow_new else 'not followed'}"
            )
        if args.subcommand == "rescan":
            for group in args.groups:
                table.rescan(group, forget=args.forget)
            groups = ", ".join(map(str, args.groups))
            if args.forget:
                return success(f"Encryption forgotten: {groups}")
            return success(f"Looked at again at the next encrypted call: {groups}")
        if args.subcommand == "talkgroups":
            rows = table.rows()
            if not rows:
                return "No talkgroups heard yet"
            return "\n".join(
                f"  TG {t.group:<6} "
                f"{'[cyan]followed[/]' if table.follows(t.group) else '[dim]not followed[/]'}"
                + "".join(f"  ALGID 0x{alg:02X} key 0x{kid:04X}" for alg, kid, _ in t.keys)
                for t in rows
            )
        raise SDRException(f"Unknown subcommand: {args.subcommand}")

    @staticmethod
    def _follow(table: SystemState, targets: list[str]) -> str:
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
