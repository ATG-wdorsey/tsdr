# TSDR

This is a fork of [floens/tsdr](https://github.com/floens/tsdr) by floens, maintained by
[ATG-wdorsey](https://github.com/ATG-wdorsey). **Version 0.2.0** (2026-10-15) adds fixes and enhancements made while
using TSDR with a PlutoSDR and an Airspy on Linux: what changed and when is in [CHANGELOG.md](CHANGELOG.md), and the
same changes as offered upstream ([floens/tsdr#2-#11](https://github.com/floens/tsdr/pulls)) in [PRs.md](PRs.md).
Branch `main` holds the latest release; each release is also a tag (`v0.2.0`). Branch `master` mirrors upstream.

TSDR is a terminal user interface for software-defined radio. It provides a
real-time spectrum display, a waterfall, audio demodulation, and a set of
protocol decoders, all driven from the keyboard.

It is aimed at users who prefer keyboard-driven, terminal-based tools and value
simplicity. The terminal UI handles tuning and visualization, while the console
exposes the full feature set through commands. This makes TSDR usable both as an
interactive radio and as a general SDR toolkit.

> **Note:** TSDR is in heavy development. Interfaces and commands may change.

![Screenshot of TSDR](docs/files/screenshot_20260604.png)

## Features

- Real-time spectrum and waterfall display, with optional Kitty graphics for
  high-frame-rate image rendering if your terminal supports it (recommended!
  see also the recommended terminals in the requirements section)
- Audio demodulators: all you can expect. WFM (with stereo and RDS), NFM, AM,
  SSB (USB/LSB), CW (Morse).
- Protocol decoders: this list is always expanding. FLEX paging (2-FSK),
  ADS-B (1090 MHz Mode S), DAB+ (AAC), DMR (with AMBE+2), TETRA (with ACELP),
  FT8/FT4 (WSJT-X), APRS (AFSK1200/AX.25), SSTV (Martin/Scottie/Robot),
  NAVTEX, ACARS.
- Multiple devices and multiple independent processing pipelines per device. It
  also connects to remote devices, such as KiwiSDR and spyserver.
- A command console with history, tab-completion, and argparse-style help.
- Frequency memories, bandplan overlays, IQ recording, and squelch.
- Runs on Linux, macOS, and Windows.

### New in 0.2.0

Measured on a PlutoSDR and an Airspy under Linux; the details are in [CHANGELOG.md](CHANGELOG.md).

- **No phantom signals from strong stations.** Demodulators and decoders now decimate in stages, so a strong station
  anywhere else in the capture can no longer fold onto the channel you are tuned to: it is held at least 81 dB down
  (as little as 0.4 dB before, at 10 MS/s). FT8/FT4 decode the upper sideband only.
- **Zero-IF radios without the center spike.** On a PlutoSDR, HackRF, LimeSDR, USRP or bladeRF the LO now sits just
  beside the channel you listen to instead of on it, keeping the DC spike, LO leakage and near-DC intermod out of the
  audio (on a Pluto+ on UHF the channel at the LO was 7.5-9.7 dB noisier). `o` / `O` put it below or above the dial.
- **No overload from signals outside the span.** The Pluto's analog filter now follows the sample rate instead of
  staying at 18 MHz: a tone a few MHz outside a 2 MS/s span overloaded the ADC from -35 dBm and lifted the noise floor
  16 dB; now there is no overload up to -20 dBm, and what aliases into the span is 24 dB lower.
- **Gap-free audio on Linux.** About 385 audio dropouts a minute with the TUI running, down to none (tested to
  10 MS/s): tsdr writes as much as PulseAudio or WASAPI reports room for.
- **Every sample, even at 10 MS/s.** SoapySDR streams are read on a native thread that Python can't pause, and the
  ~120 ms garbage-collection pauses are gone: a PlutoSDR at 10 MS/s delivers the full 10.00 MS/s inside the TUI
  (9.4-9.8 before), and holding a tuning key loses nothing.
- **Your radio's real limits.** SoapySDR radios use their actual frequency range, sample rates (such as the Airspy's
  two fixed rates) and gain step. Lost samples are never silent: driver overflows are logged, and the screen says when
  a radio delivers less than its rate and suggests a lower one.
- **A true spectrum when zoomed.** Each column averages the FFT bins it covers: one zoom step showed 7.4 dB of
  column-to-column jitter and read up to 9.5 dB high; now the jitter is 2.0 dB.
- **Polish.** Kitty image mode no longer crashes (SIGBUS) or leaves frames in `/dev/shm`; the radio always matches the
  dial after a held tuning key; a shortcut can't exit the app; click-to-tune snaps to the channel raster; modes are
  accepted in any case; the console takes keys after a panel click. The version shows in the Stats panel's title, and
  `tsdr --version` prints it.


## Requirements

- Python 3.14 or newer.
- A supported SDR device, or a device to connect to over the internet.
- A modern, full-featured terminal emulator. [Ghostty](https://ghostty.org) is
  recommended on Linux and macOS for its broad protocol support. Image mode uses
  the Kitty graphics protocol; Ghostty and [Rio](https://rioterm.com/) render it
  best. On Windows, use Rio or a recent
  [Windows Terminal](https://github.com/microsoft/terminal) (Terminal Preview
  preferred), where only Rio supports Kitty images.

Supported device types: `rtltcp`, `rtlsdr`, `spyserver`, `soapy` (any
SoapySDR-compatible device), `iq-file`.

## Installation

TSDR is not published on PyPI. Install this fork directly from GitHub with
[uv](https://docs.astral.sh/uv/), either the latest release (`main`) or a given one (`@v0.2.0`):

```
uv tool install --from git+https://github.com/ATG-wdorsey/tsdr@main tsdr
```

The RTL-SDR USB support (the `rtlsdr` device type) requires the optional
`rtlsdr` extra:

```
uv tool install --from git+https://github.com/ATG-wdorsey/tsdr@main "tsdr[rtlsdr]"
```

To update, run the same command with `--reinstall`. The original project installs with
`git+https://github.com/floens/tsdr`.

### SoapySDR support

The `soapy` device type uses [SoapySDR](https://github.com/pothosware/SoapySDR),
which is not a pip package. Its Python bindings ship with the system C++
library, so it is installed through the OS package manager, along with the
driver module for the target hardware. TSDR then detects the system installation
automatically by probing the usual Homebrew and apt site-package locations, so
the isolated uv/pipx environment finds it without extra configuration.

On macOS, the Homebrew formula builds the Python bindings; driver modules come
from the Pothos tap:

```
brew install soapysdr
brew tap pothosware/homebrew-pothos
brew install soapyrtlsdr      # or soapyhackrf, soapyairspy, soapyremote, ...
```

On Debian/Ubuntu:

```
sudo apt install python3-soapysdr soapysdr-tools
sudo apt install soapysdr-module-rtlsdr   # or -hackrf, -airspy, -remote, ...
```

Run `soapy probe` in the console to enumerate the devices SoapySDR can see.

TSDR also sets the analog filter ahead of the radio's converter when the driver
leaves it wide. The Pluto's stays at 18 MHz whatever the sample rate, so a strong
signal a few MHz outside a 2 MS/s span overloads the converter from about -35 dBm
at gain 40 and aliases into the span. On the Pluto TSDR keeps the filter at 0.8 x
the sample rate; other drivers keep their own. `config --analog-filter` takes
`track`, `off` (the driver's own setting), `auto`, or a fixed width such as
`1.5M`, and `--analog-filter-fraction` sets the fraction. The Stats panel shows
the width the driver reports.

### Checking your terminal and configuration

After installing, run the built-in diagnostic to verify that your terminal and
environment support everything TSDR needs (notably the graphics protocol used by
image mode):

```
tsdr doctor
```

Use `tsdr doctor --check` for a non-interactive report that exits non-zero on
failure, or `tsdr doctor --json` for machine-readable output.

## Usage

Start the application:

```
tsdr
```

You can run commands automatically on startup with `--exec` (repeatable), for
example to add and start a device:

```
tsdr -e "add rtl0 --type rtltcp --host 192.168.1.10" -e "start rtl0"
```

Other entrypoint flags:

- `-e`, `--exec COMMAND`: run a console command on startup (may be repeated).
- `--headless`: run without the TUI; logs go to stdout and commands are read
  from stdin.
- `-v`, `--verbose`: enable debug logging.
- `--version`: print the version and exit.

### The console

Press `` ` `` (backtick) to focus the command console. The console behaves like
a shell:

- Type a command and press Enter to run it.
- Press `Tab` / `Shift+Tab` to cycle autocompletion (command names, device IDs,
  flag values).
- Use the arrow keys or `Ctrl+R` to browse and search history.
- Press `` ` `` or `Escape` to return focus to the spectrum.

### Discovering commands

Run `help` to list all commands with a one-line description of each. To see the
options for a single command, run it with no arguments or pass `-h`, which prints
its argparse usage and flags. For example:

```
help
demod -h
config -h
```

A typical session: add a device, start it, tune, and pick a demodulator.

```
add rtl0 --type rtlsdr
start rtl0
f 100.1M
demod wfm
```

Common commands include `add`, `remove`, `start`, `stop`, `list`, `use`, and
`config` (device management); `f` (tune), `demod` (select a demodulator or
decoder), `squelch`, `denoise`, and `audio` (audio); `search` (find active
channels); `memory` and `bandplan` (frequency presets); `record` (IQ capture); and `pipeline` (inspect or modify
processing stages). See `help` for the full list.

### Keyboard controls

When the console is unfocused, the spectrum responds to the keyboard directly.

Tuning:

- `f`: type a frequency on the status line (`Frequency: `), with the console's
  line editing and its own history (`↑` / `↓`); Enter tunes, Escape leaves it.
  A bare number is MHz (`146.52`); `k`, `M` and `G` (either case) set the unit
  (`7074k`).
- `←` / `→`: tune down / up one step.
- `Shift+←` / `Shift+→`: tune one coarse step.
- `Alt+←` / `Ctrl+←`, `Alt+→` / `Ctrl+→`: tune one fine step.
- `[` / `]`: jump to the previous / next signal or memory within the captured
  band. A signal is what SEARCH would stop on: channels on the tuning-step grid
  at or above the squelch, taken once each at the strongest channel (without a
  demodulator, spectrum peaks). They stop at the band's edge, so in free
  tuning they never move the LO. While SEARCH runs they stay within its range
  and keep it running (see below).
- `s` / `S`: cycle the tuning step forward / backward through the steps that fit the mode,
  starting from auto (the mode's own step): NFM and the FM decoders 2.5, 5, 6.25, 10,
  12.5, 15, 20, 25, 50 and 100 kHz; USB/LSB and the decoders on them 10 Hz to 10 kHz; CW
  10 Hz to 5 kHz; AM 1 to 100 kHz (9 and 10 kHz for broadcast); WFM 50, 100 and 200 kHz.
  Each mode keeps its own step (5 kHz in NFM stays 5 kHz after a detour to USB),
  remembered across restarts; a mode you never set steps by auto (12.5 kHz for NFM,
  100 kHz for WFM).
- Click the spectrum to tune there. The click snaps to the mode's channel raster
  (12.5 kHz for NFM, 100 kHz for WFM), or to the tuning step when that is finer.
- `c`: toggle center / free tuning (center: hardware follows every dial move;
  free: DSP offset until the band edge; entering center recenters).
- `o` / `O`: put the LO below / above the dial while demodulating; the same key
  again removes the offset. Zero-IF radios (Pluto, HackRF, LimeSDR, USRP,
  bladeRF) keep the LO below the dial by default, one channel width away and at
  least 50 kHz, so their center spike and intermod products stay out of the
  channel. In center mode the spectrum is trimmed by twice the offset to keep
  the dial centered. `config --lo-offset` and `--lo-offset-size` set it from the
  console, and the Stats panel shows where the LO is.

Search (find activity within the captured band):

- `/`: start / stop SEARCH. It switches to free tuning and turns the squelch on,
  then puts the dial on the strongest channel that stays above the trigger for
  100 ms (channels on the tuning-step grid, within the captured band; the
  range is shaded on the spectrum). Its grid is never finer than 1 kHz, or than
  the FFT resolves: with the tuning step at 10 Hz it searches 1 kHz steps. Nor
  is it coarser than the channel, which would leave stations between grid
  points unseen: a 100 kHz step gives way to the mode's automatic step (12.5 kHz
  for NFM), and that mode's tuning step goes to auto. The squelch holds the channel:
  it opens at its line (`SQ`) and shuts only when the channel drops 3 dB under
  it (the `HY` line); SEARCH moves on only after the squelch has stayed shut for the hold time
  (2 s), so replies are heard, and between hits it stays on the last
  channel. `[` / `]` move it on to the previous / next signal within its range
  (skipping locked-out channels) without stopping it, so you can leave a channel
  before its hold ends; other tuning by hand, turning the squelch off (`Ctrl+U`)
  or stopping the radio stops it.
  For narrowband modes only (not WFM).
- `{` / `}`: put SEARCH's lower / upper edge at the dial (without edges it
  searches the whole captured band; an edge outside the band is ignored, and
  one on the wrong side of the other edge is refused).
- `x`: lock out the current channel for this session. SEARCH stops only on a
  signal's center (a channel stronger than its neighbors), so the edges of a
  locked-out signal, which a strong one pushes over the trigger 10-15 kHz
  away, are passed over too, while a station of its own beside it is not.
- `X`: lock out the current channel for good (a memory tagged `lockout`).
- `Ctrl+X`: unlock the current channel when it is locked out (deleting its
  lockout memory; `X` there makes it again), or else drop all session lockouts.
  While SEARCH runs, the tuner shows how many lockouts are in force (`2 L/O`).
- `v`: VSQ, the voice squelch, on / off while SEARCH runs (later runs keep
  it; `search --vsq` / `--no-vsq`). A detector listens for speech in each
  transmission SEARCH stops on, whether or not the denoiser is on; one without
  speech in its first 3 s (data, digital voice, tones, SSTV, a CW ID; a quiet
  carrier gets 5 s, for a late talker) is muted and the
  hold counts down (`VSQ` in place of `HOLD`), time to lock it out with `x`;
  then SEARCH moves on and leaves that channel until its signal drops. Speech
  reopens it. The meter's squelch reads `VSQ` while it runs; the Stats panel
  shows its verdict and the detector's last label.

`search on --low 144M --high 148M --threshold -60 --hold 2 --dwell 30` sets the
edges, the trigger (dBFS within the channel, as the squelch measures it;
by default the squelch itself, following it, so `u` / `U` set both), the
hold time and the most time on one
active channel (`--dwell 0`: unlimited, the default); `--full` forgets the
edges. `search` shows the settings, `search off` stops, `search lockout
[--permanent]` locks out the current channel, and `search unlock` drops this
session's lockouts.

Reception:

- `↑` / `↓`: adjust channel bandwidth.
- `Alt+↑` / `Ctrl+↑`, `Alt+↓` / `Ctrl+↓`: adjust channel bandwidth finely.
- `g` / `G`: decrease / increase RF gain.
- `Ctrl+G`: toggle AGC.
- `u` / `U`: lower / raise the squelch threshold 1 dB.
- `Ctrl+U`: disable squelch.
- `Shift+↑` / `Shift+↓`: decrease / increase volume.
- `n`: toggle RNNoise denoising (`DN` beside the volume).
- `Space`: pause / resume the focused radio. Paused, the radio keeps streaming
  (resuming is instant, with none of a restart's delays) and tsdr drops what it
  reads: audio, spectrum and waterfall, decoders, SEARCH and recording all stop
  where they are (`PAUSED` in the tuner). SEARCH stays on, its hold frozen. Keys
  and clicks that operate the radio are ignored until Space resumes; the
  console, the panels and the display keys (`i`, `k`/`j`, `h`/`l`, `H`/`L`,
  `C`) still work. On a stopped radio Space starts it; `stop` stops it.
- `d` then a second key: switch demodulator (`w` WFM, `n` NFM, `a` AM,
  `u` USB, `l` LSB, `c` CW, `o` off).
- `D` then a second key: switch decoder (`a` ADS-B, `b` DAB, `c` ACARS, `d`
  DMR, `f` FLEX, `k` FSK, `n` NAVTEX, `p` APRS, `r` RTTY, `s` SSTV, `t` TETRA,
  `4` FT4, `8` FT8, `o` off). Both prompts show each choice's key in place,
  bright and in upper case, on a second line when the window is narrow; a key
  that is not a choice cancels. SSTV decodes the mode you are in instead of
  switching it: NFM for SSTV on 2 m (145.600), USB on HF or 144 MHz SSB; the
  bandwidth, squelch and audio stay yours, and the tuner reads `FM + SSTV`.
  With no audio mode on, it starts in NFM above 30 MHz and USB below. `D o`
  then turns off only the decoder, and `d` keeps it across NFM, USB, LSB and AM
  (`demod sstv` in the console does the same).

Display:

- `i`: toggle image mode.
- `k` / `j`: zoom the spectrum in / out (mouse: `Shift+scroll`; plain scroll
  tunes, or pans the view in free mode when zoomed).
- `C`: re-center a panned view on the dial.
- `h` / `l`: raise / lower the noise floor (spectrum dB minimum).
- `H` / `L`: raise / lower the ceiling (spectrum dB maximum).

The spectrum has a dB grid (every 10 dB, or 5 dB in a narrow window), labeled
in a column on the left; the waterfall's column is its color key, strongest at
the top. Each label is in the color a signal at that level takes on the
waterfall (gray for the noise floor's dark blues). With the squelch on, a dashed orange line (`SQ`) marks the squelch:
the squelch opens, and SEARCH stops on a channel, when the channel reaches it.
While SEARCH runs, a dimmer dashed line (`HY`) 3 dB below it over the search
range marks where the squelch shuts again (a held channel has to drop under
it), and a yellow line (`TR`) marks a trigger set by hand. The trace is power per FFT bin while the squelch measures the whole
channel, so the lines sit where noise at those levels shows: a squelch set
just above the noise floor is drawn just above it. A narrow carrier, all in
one bin, has to rise further above the line (about 14 dB for NFM at 10 MS/s)
to open the squelch.

Memories and panels:

- `m`: save a memory at the current frequency.
- `M`: edit the nearest memory.
- `Ctrl+M`: delete the nearest memory (press `y` to confirm).
- `1` to `9`: toggle the dockable panel mapped to that key.
- `?`: show / hide the list of keys (a panel, like the numbered ones).
- `Alt+1` to `Alt+9`: move the mapped panel to the next edge (left, bottom, right).
- `Ctrl+1` to `Ctrl+9`: recall a band-stack slot; `Ctrl+0`: swap the A/B band stack.

When the console is focused:

- `` ` `` or `Escape`: return focus to the spectrum (`Escape` first dismisses
  an open autocomplete menu).
- `Tab` / `Shift+Tab`: open and cycle the autocomplete menu forward / backward.
- `↑` / `Ctrl+P`, `↓` / `Ctrl+N`: previous / next history entry.
- `Ctrl+R`: search history.
- `Ctrl+L`: clear the console.

## Configuration and data

Logs and preferences are stored in a per-user directory managed by
[platformdirs](https://pypi.org/project/platformdirs/):

- Linux: `~/.config/tsdr` (or `$XDG_CONFIG_HOME/tsdr`)
- macOS: `~/Library/Application Support/tsdr`
- Windows: `%APPDATA%\tsdr`

The `paths` command prints the exact location and its contents. Some
configuration (such as the last device and demodulator) is persisted there and
restored on startup.

## Design

The architecture, goals, and non-goals are described in [DESIGN.md](DESIGN.md).
At a high level, a `core` layer owns all SDR logic (devices, workers, pipelines,
the event bus), and a `tui` layer renders it. It is written in Python so it is easy
to read, modify, and experiment with, with hot DSP paths accelerated by NumPy
and Numba.

## Development

From floens, the original author: Parts of TSDR were written with the help of LLMs. The development and porting of
complicated, digital decoders is aided with LLMs, building on top of the work of
many authors and programs in this space. This program is therefore licensed
under the GPL (as is most of my software).

## Acknowledgments

TSDR includes code derived from or inspired by several other projects. See
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).

## License

TSDR is licensed under the GNU General Public License, version 2 or later. See
[COPYING](COPYING). TSDR is by floens; this fork's changes are by their authors as the git log records.
