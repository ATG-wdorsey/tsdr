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
decoder), `squelch`, `denoise`, and `audio` (audio); `memory` and `bandplan`
(frequency presets); `record` (IQ capture); and `pipeline` (inspect or modify
processing stages). See `help` for the full list.

### Keyboard controls

When the console is unfocused, the spectrum responds to the keyboard directly.

Tuning:

- `←` / `→`: tune down / up one step.
- `Shift+←` / `Shift+→`: tune one coarse step.
- `Alt+←` / `Ctrl+←`, `Alt+→` / `Ctrl+→`: tune one fine step.
- `[` / `]`: jump to the previous / next tuning target (memory or band).
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
  console, and the Stats panel shows where the LO is.

Reception:

- `↑` / `↓`: adjust channel bandwidth.
- `Alt+↑` / `Ctrl+↑`, `Alt+↓` / `Ctrl+↓`: adjust channel bandwidth finely.
- `g` / `G`: decrease / increase RF gain.
- `Ctrl+G`: toggle AGC.
- `u` / `U`: lower / raise the squelch threshold.
- `Ctrl+U`: disable squelch.
- `Shift+↑` / `Shift+↓`: decrease / increase volume.
- `n`: toggle RNNoise denoising.
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

Memories and panels:

- `m`: save a memory at the current frequency.
- `M`: edit the nearest memory.
- `Ctrl+M`: delete the nearest memory (press `y` to confirm).
- `1` to `9`: toggle the dockable panel mapped to that key.
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
