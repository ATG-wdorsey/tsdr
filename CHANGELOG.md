# Changelog

Notable changes to this fork of [floens/tsdr](https://github.com/floens/tsdr), newest first. Each release is a
git tag (`v0.2.0`, ...); `main` on GitHub always holds the latest release. Version 0.1.0 is floens/tsdr as of
[`6b52133`](https://github.com/floens/tsdr/commit/6b52133) (2026-09-12).

The changes in 0.2.0 were also offered upstream as pull requests; their full descriptions (why, cause, change,
testing) are in [PRs.md](PRs.md).

## [0.2.0] - 2026-10-15

The first release of the fork: fixes and enhancements made while using TSDR with a PlutoSDR and an Airspy on Linux.

### Added

- The version is shown in the Stats panel's title (`Stats (TSDR v0.2.0)`), and `tsdr --version` prints it.
- SoapySDR radios use their real limits: the driver's frequency range, its fixed sample rates (such as the Airspy's) and
  its gain step. ([#6](https://github.com/floens/tsdr/pull/6))
- Lost samples are reported: driver overflows are counted and logged, and a warning suggests a lower rate when a radio
  delivers well below the rate asked for. ([#6](https://github.com/floens/tsdr/pull/6))
- SoapySDR streams are read on a native thread, so pauses in Python can't drop radio samples, and holding a tuning key
  loses nothing. ([#8](https://github.com/floens/tsdr/pull/8))
- Zero-IF radios (PlutoSDR, HackRF and similar) keep the LO beside the demodulated channel instead of on it (on a Pluto+
  on UHF the channel at the LO was 7.5-9.7 dB noisier), keeping the DC spike, LO leakage and near-DC intermod out of
  what you hear: `o` / `O` put it below or above the dial, `config --lo-offset` and `--lo-offset-size` set it; a
  fixed-width decoder's own width sizes the offset. ([#9](https://github.com/floens/tsdr/pull/9))
- The Pluto's analog filter follows the sample rate (0.8 x) instead of staying at 18 MHz, so strong signals outside the
  span no longer overload the converter (at 2 MS/s and gain 40, a tone a few MHz outside the span overloaded it from -35
  dBm; now not up to -20 dBm, and aliases are 24 dB lower): `config --analog-filter`. The Stats panel shows the LO and
  the filter's width. ([#9](https://github.com/floens/tsdr/pull/9))

### Fixed

- No phantom signals while tuning: strong stations elsewhere in the capture no longer fold onto the tuned channel (at 10
  MS/s they were rejected by 0.4 dB; now by at least 81 dB). The audio demodulators and the protocol decoders decimate
  in stages; the frequency shift's rotator comes from a table. ([#7](https://github.com/floens/tsdr/pull/7))
- WSJT modes (FT8/FT4) demodulate the upper sideband only. ([#7](https://github.com/floens/tsdr/pull/7))
- Clean audio on Linux with the TUI running: PulseAudio and WASAPI get a real device buffer, and writes are paced by the
  room the device reports, not a clock estimate: about 385 dropouts a minute down to none (tested to 10 MS/s); the
  underrun counter counts real underruns. ([#3](https://github.com/floens/tsdr/pull/3))
- Garbage-collection pauses that stopped every thread for about 120 ms, dropping radio samples at high sample rates, are
  kept short (a PlutoSDR at 10 MS/s delivered 9.80 MS/s in the TUI, now 10.00); the tuner's device line is repainted
  only when it changes. ([#5](https://github.com/floens/tsdr/pull/5))
- Kitty no longer crashes (SIGBUS) in image mode, and frames no longer pile up in `/dev/shm`: each frame gets its own
  shared-memory object. ([#4](https://github.com/floens/tsdr/pull/4))
- Everyday glitches ([#2](https://github.com/floens/tsdr/pull/2)):
  - after holding a tuning key the radio could stay on a different frequency (or filter width) than the dial showed;
  - an out-of-range keyboard shortcut exited the app;
  - a radio that failed to open showed as running;
  - click-to-tune landed on whole megahertz with coarse steps (clicks now snap to the mode's channel raster);
  - `demod wfm` was rejected (modes were case-sensitive).
- A steady spectrum when zoomed or offset: each column averages the FFT bins it covers instead of showing one (one zoom
  step: 7.4 dB of jitter between columns and up to 9.5 dB high, now 2.0 dB of jitter).
  ([#10](https://github.com/floens/tsdr/pull/10))
- After a click on a panel, `` ` `` opened the console but the keys didn't reach it; a change made by the automatic gain
  could leave the display on the old settings. ([#11](https://github.com/floens/tsdr/pull/11))

### Documentation

- README rewritten for this release; [PRs.md](PRs.md) holds the pull requests offered upstream; this changelog.
