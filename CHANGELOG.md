# Changelog

Notable changes to this fork of [floens/tsdr](https://github.com/floens/tsdr), newest first. Each release is a
git tag (`v0.2.0`, ...); `main` on GitHub always holds the latest release. Version 0.1.0 is floens/tsdr as of
[`6b52133`](https://github.com/floens/tsdr/commit/6b52133) (2026-09-12).

The changes in 0.2.0 were also offered upstream as pull requests; their full descriptions (why, cause, change,
testing) are in [PRs.md](PRs.md).

## [0.3.0] - 2026-10-16

SEARCH and its voice squelch, DMR's two timeslots, SSTV on any mode's audio, new keys and a readable spectrum: the work
of 2026-10-12 to 2026-10-15 that has been in daily use since.

### Added

- **SEARCH** (`/`, the `search` command) finds activity within the capture and stops on it, holding while the squelch is
  open and for the hold time after (2 s). Edges at the dial (`{` / `}`) and a shaded range; a trigger 3 dB above the
  squelch, or set by hand (`TR`); lockouts for the session (`x`, `Ctrl+X` unlocks; the tuner and `search` list them) or
  lasting ones as memories; `[` / `]` move a running SEARCH on. Its grid follows the tuning step, never finer than 1 kHz
  or an FFT bin nor coarser than the channel; it jumps on a channel's power averaged over 0.1 s (with the noise 1 dB
  under the squelch line it spent 95 % of its time on silent channels in NFM, now none) and stops only on a signal's
  center; decoder modes get a level squelch.
- **VSQ**, a voice squelch for SEARCH (`v`, `search --vsq`): a small neural network, trained on real over-the-air audio,
  lets SEARCH leave a transmission with no speech by 3 s (5 s for a quiet carrier); a short squelch fade doesn't end a
  transmission.
- **DMR**: hear TS1, TS2, the first call (auto) or both (TS1 left, TS2 right): `t` / `T`, `demod dmr --slot`. Calls
  followed through a late entry or a missed sync in both slots (both could go silent for the rest of a call in 5 of the
  6 ways the slots' superframes align); repeaters off frequency (one ~450 Hz off: 351 of 357 frames without a bit error,
  53 before); bursts taken by timing checked; the speech level set per call as the reference decoders do, checked
  against mbelib; a matched filter and a symbol constellation.
- **SSTV** decodes the current mode's audio (NFM on 2 m, USB on HF) instead of switching to USB; joins a picture already
  under way from its line syncs and takes the mistune off; a picture ends with its sender, survives fades and lies
  straight on a line fitted to its syncs (the next picture starts from its VIS 24 of 24 times at 6 and 12 dB SNR, 6-8
  before).
- New keys: `f` types a frequency on the status line (line editing, its own history, k/M/G suffixes); `D` picks a
  decoder; `?` lists the keys; Space pauses and resumes the radio instead of stopping it.
- `[` / `]` jump between signals and memories within the captured band (in free tuning they never move the LO).
- A dB grid with labels on the spectrum, the squelch (`SQ`), hysteresis (`HY`) and trigger (`TR`) lines, a color key for
  the waterfall, and a cold-to-hot waterfall colormap.
- Each mode keeps its own tuning step, chosen from steps that fit the mode and remembered across restarts.
- ADS-B and DAB decode at the radio's own capture rate.
- NFM filters CTCSS tones out of the audio (they carried 85-98 % of its power on local repeaters; now 57 dB down). WFM
  plays stereo at the mono level (it was 6 dB quieter), detects the pilot against the guard band's noise, keeps the 38
  kHz carrier in phase, and remembers its de-emphasis setting.
- `tsdr.log` keeps the last five logs.
- The voice detector's training data (MUSAN noise) is credited in ACKNOWLEDGMENTS.md.

### Changed

- The squelch opens at its line and closes 3 dB under it, and logs each change; `u` / `U` move it 1 dB a press; it is
  kept across demodulator changes and across turning demodulation off and on.
- The mouse wheel tunes one tuning step, up for up; another demodulator starts at its own bandwidth, and a fixed-width
  decoder leaves the bandwidth setting alone.
- The denoiser passes WFM through while its pilot is unlocked. OpenBLAS runs on one thread (the voice detector's matrix
  products woke eleven spinning threads).

### Fixed

- A started device comes into view when the focused one is stopped; a SoapySDR stream starts on its first read, after
  the radio is tuned (the Airspy stopped streaming otherwise); a failed wheel tune shows an error instead of ending the
  app.
- A huge typed frequency, or an LO at or below 0 Hz, is refused instead of ending the app; a damaged voice model makes
  VSQ unavailable instead of ending the app; a history file that isn't UTF-8 still reads; settings changed while a
  device is stopped reach the processing stages when it starts.
- The polyphase resampler (ADS-B and DAB rate conversion) meets the newest sample with each branch's first tap; DMR
  counts continuation bursts per timeslot.

### Known issues

- SSTV: a picture joined while tuned more than about 100 Hz low can lock on its black porches; APRS, RTTY or PSK31
  can start a false picture.
- DMR: a second repeater on the same frequency goes unheard; a radio in direct mode is heard only part of the time;
  a memory recalled while in DMR keeps the current timeslot.
- FT8/FT4 decode nothing after a pause until the mode is picked again.
- ADS-B and DAB keep their capture rate after a sample-rate change.
- The first use of a mode after an install pauses for 1-4 s while its code compiles.

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
