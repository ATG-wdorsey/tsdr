# TSDR

> **About this fork.** This fork of [floens/tsdr](https://github.com/floens/tsdr)
> carries fixes and enhancements made while using TSDR with a PlutoSDR and an
> Airspy on Linux, all offered upstream as pull requests. What you'll notice:
>
> **Fixes**
> - **No phantom signals while tuning.** Strong stations elsewhere in the
>   capture no longer fold onto the channel you are tuned to (at 10 MS/s they
>   were rejected by 0.4 dB; now by at least 81 dB). WSJT modes decode the
>   upper sideband only.
>   ([floens/tsdr#7](https://github.com/floens/tsdr/pull/7))
> - **Clean audio on Linux with the TUI running.** No more dropouts several
>   times a second, and the underrun counter counts real underruns.
>   ([#3](https://github.com/floens/tsdr/pull/3))
> - **Smooth capture at high sample rates.** Garbage-collection pauses that
>   stopped every thread for ~120 ms, dropping radio samples, are kept short.
>   ([#5](https://github.com/floens/tsdr/pull/5))
> - **Kitty no longer crashes** (SIGBUS) in image mode, and frames no longer
>   pile up in `/dev/shm`. ([#4](https://github.com/floens/tsdr/pull/4))
> - **Everyday glitches.** After holding a tuning key the radio could stay on
>   a different frequency than the dial showed; an out-of-range shortcut
>   exited the app; a radio that failed to open showed as running;
>   click-to-tune landed on whole megahertz with coarse steps; `demod wfm`
>   was rejected (modes were case-sensitive).
>   ([#2](https://github.com/floens/tsdr/pull/2))
> - **A steady spectrum when zoomed or offset.** Zooming in, or an LO offset,
>   no longer makes the trace noisy and several dB too high: each column
>   averages the FFT bins it covers instead of showing just one.
>   ([#10](https://github.com/floens/tsdr/pull/10))
>
> **Enhancements**
> - **SoapySDR radios use their real limits:** frequency range, fixed sample
>   rates (such as the Airspy's) and gain step. Lost samples are reported:
>   driver overflows are logged, and a warning suggests a lower rate when a
>   radio can't deliver the one asked for.
>   ([#6](https://github.com/floens/tsdr/pull/6))
> - **Capture outside Python.** SoapySDR streams are read on a native thread,
>   so pauses in Python can't drop radio samples, and holding a tuning key
>   loses nothing. ([#8](https://github.com/floens/tsdr/pull/8))
> - **Zero-IF radios (PlutoSDR, HackRF and similar).** While you demodulate,
>   the LO sits beside the channel instead of on it, keeping the DC spike, LO
>   leakage and near-DC intermod out of what you hear: `o`/`O` put it below or
>   above the dial (again: no offset), or `config --lo-offset` and
>   `--lo-offset-size`. The Pluto's analog filter follows the sample rate
>   (0.8x) instead of staying at 18 MHz, so strong signals outside the span no
>   longer overload the ADC: `config --analog-filter`.
>   ([#9](https://github.com/floens/tsdr/pull/9))
>
> Install this fork with
> `uv tool install --from git+https://github.com/ATG-wdorsey/tsdr@main tsdr`.
> Branch `main` is this fork; `master` mirrors upstream. The git log records
> who changed what, and when.

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
- SSB (USB/LSB), CW (Morse).
- Protocol decoders: this list is always expanding. FLEX paging (2-FSK),
  ADS-B (1090 MHz Mode S), DAB+ (AAC), DMR (with AMBE+2), TETRA (with ACELP),
  FT8/FT4 (WSJT-X), APRS (AFSK1200/AX.25), SSTV (Martin/Scottie/Robot),
  NAVTEX, ACARS.
- Multiple devices and multiple independent processing pipelines per device. It
  also connects to remote devices, such as KiwiSDR and spyserver.
- A command console with history, tab-completion, and argparse-style help.
- Frequency memories, bandplan overlays, IQ recording, and squelch.
- Runs on Linux, macOS, and Windows.

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

TSDR is not published on PyPI (yet). Commended to install it directly from
GitHub with [uv](https://docs.astral.sh/uv/):

```
uv tool install --from git+https://github.com/floens/tsdr tsdr
```

The RTL-SDR USB support (the `rtlsdr` device type) requires the optional
`rtlsdr` extra:

```
uv tool install --from git+https://github.com/floens/tsdr "tsdr[rtlsdr]"
```

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

- `←` / `→`: tune down / up one step.
- `Shift+←` / `Shift+→`: tune one coarse step.
- `Alt+←` / `Ctrl+←`, `Alt+→` / `Ctrl+→`: tune one fine step.
- `[` / `]`: jump to the previous / next signal or memory within the captured
  band. A signal is what SEARCH would stop on: channels on the tuning-step grid
  at or above the squelch, taken once each at the strongest channel (without a
  demodulator, spectrum peaks). They stop at the band's edge, so in free
  tuning they never move the LO. While SEARCH runs they stay within its range
  and keep it running (see below).
- `s` / `S`: cycle the tuning step size forward / backward.
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
  console.

Search (find activity within the captured band):

- `/`: start / stop SEARCH. It switches to free tuning and turns the squelch on,
  then puts the dial on the strongest channel that stays above the trigger for
  100 ms (channels on the tuning-step grid, within the captured band; the
  range is shaded on the spectrum). Its grid is never finer than 1 kHz, or than
  the FFT resolves: with the tuning step at 10 Hz it searches 1 kHz steps. The squelch holds the channel:
  SEARCH moves on only after the squelch has stayed shut for the hold time
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
by default 3 dB above the squelch, following it, so `u` / `U` set both), the
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
- `Space`: start / stop the focused device.
- `d` then a second key: switch demodulator (`w` WFM, `n` NFM, `a` AM,
  `u` USB, `l` LSB, `c` CW, `o` off).

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
waterfall (gray for the noise floor's dark blues). With the squelch on, a dashed orange line (`SQ`) marks the squelch,
and while SEARCH runs a yellow line (`TR`) over the search range marks its
trigger. The trace is power per FFT bin while the squelch measures the whole
channel, so the lines sit where noise at those levels shows: a squelch set
just above the noise floor is drawn just above it. A narrow carrier, all in
one bin, has to rise further above the line (about 14 dB for NFM at 10 MS/s)
to open the squelch.

Memories and panels:

- `m`: save a memory at the current frequency.
- `M`: edit the nearest memory.
- `Ctrl+M`: delete the nearest memory (press `y` to confirm).
- `1` to `9`: toggle the dockable panel mapped to that key.
- `Ctrl+S`: toggle the stats panel; `Ctrl+P`: toggle the performance panel.
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
the event bus), and a `tui` layer renders it. s written in Python so it is easy
to read, modify, and experiment with, with hot DSP paths accelerated by NumPy
and Numba.

## Development

Parts of TSDR were written with the help of LLMs. The development and porting of
complicated, digital decoders is aided with LLMs, building on top of the work of
many authors and programs in this space. This program is therefore licensed
under the GPL (as is most of my software).

## Acknowledgments

TSDR includes code derived from or inspired by several other projects. See
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).

## License

TSDR is licensed under the GNU General Public License, version 2 or later. See
[COPYING](COPYING).
