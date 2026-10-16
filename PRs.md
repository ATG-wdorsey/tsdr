# Pull requests offered upstream

The changes in 0.2.0 (see [CHANGELOG.md](CHANGELOG.md)) were offered to [floens/tsdr](https://github.com/floens/tsdr)
as the ten pull requests below, opened 2026-10-11 to 2026-10-14. As of 2026-10-15 all are open, with no comments or
reviews yet.

| PR | Title | Opened | Branch |
|---|---|---|---|
| [#2](https://github.com/floens/tsdr/pull/2) | Small fixes: demod mode case, shortcut errors, failed opens, click snapping, stale configs | 2026-10-11 | `pr/small-fixes` |
| [#3](https://github.com/floens/tsdr/pull/3) | Audio: real device buffer and paced writes on PulseAudio/WASAPI (fixes Linux dropouts) | 2026-10-11 | `pr/audio-device-buffer` |
| [#4](https://github.com/floens/tsdr/pull/4) | Kitty graphics: a shared-memory object per frame (fixes SIGBUS), cleaned up | 2026-10-11 | `pr/kitty-shm-frames` |
| [#5](https://github.com/floens/tsdr/pull/5) | TUI: keep garbage-collection pauses short | 2026-10-11 | `pr/gc-pauses` |
| [#6](https://github.com/floens/tsdr/pull/6) | SoapySDR: report driver capabilities, count overflows, warn on low throughput | 2026-10-11 | `pr/soapy-capabilities` |
| [#7](https://github.com/floens/tsdr/pull/7) | DSP: multistage decimation with channel selection; WSJT upper sideband | 2026-10-11 | `pr/multistage-decimation` |
| [#8](https://github.com/floens/tsdr/pull/8) | SoapySDR: read the stream on a native thread | 2026-10-13 | `pr/soapy-native-reader` |
| [#9](https://github.com/floens/tsdr/pull/9) | Zero-IF radios: keep the LO beside the channel; track the Pluto's analog filter | 2026-10-13 | `pr/zero-if` |
| [#10](https://github.com/floens/tsdr/pull/10) | Spectrum: average the FFT bins under each column of a zoomed view | 2026-10-13 | `pr/spectrum-projection` |
| [#11](https://github.com/floens/tsdr/pull/11) | TUI: the console takes the keyboard after a panel click; config changes from the pipeline reach the UI | 2026-10-14 | `pr/tui-fixes` |

`pr/soapy-native-reader` (#8) builds on `pr/soapy-capabilities` (#6) and carries its three commits too.

The descriptions as submitted follow.

## #2: Small fixes: demod mode case, shortcut errors, failed opens, click snapping, stale configs

[floens/tsdr#2](https://github.com/floens/tsdr/pull/2), opened 2026-10-11, branch `pr/small-fixes`.

### Why
Five independent bugs that each make tsdr misbehave in everyday use:
- A held arrow key or the scroll wheel could leave the radio on a different frequency than the dial shows (hardware at 100.007 MHz with the dial at 100.040 MHz), or a filter at a different width than configured, until some unrelated setting changed: you hear and see the wrong thing with no indication why.
- An out-of-range tune from a keyboard shortcut exited the application.
- A radio that failed to open was shown as running, and `start`/`remove` were refused until a manual `stop`.
- With a coarse tuning step, every click on the spectrum landed on the same whole megahertz, so click-to-tune didn't work; a click on the dial itself also overwrote the A/B slot.
- `demod wfm` was rejected because modes were case-sensitive.

### Change
One commit each:
- Config bursts keep the newest snapshot on queue overflow; publication is serialized so concurrent producers arrive in order; the pipeline worker applies the newest pending snapshot of each type.
- Key dispatch catches `SDRException` and shows it, like commands and mouse gestures; an A/B recall outside the device range is refused.
- A failed open moves the device to ERROR (under a state lock); start/remove/reconfigure reap the leftover workers.
- Clicks snap to the mode's channel raster, or the tuning step when finer; a no-op click is not a retune.
- Demod modes are normalized to upper case before validation; choice completion is case-insensitive.

### Testing
New tests for each fix (each fails on the old code). Full suite on this branch: 1413 passed; pre-commit clean.

## #3: Audio: real device buffer and paced writes on PulseAudio/WASAPI (fixes Linux dropouts)

[floens/tsdr#3](https://github.com/floens/tsdr/pull/3), opened 2026-10-11, branch `pr/audio-device-buffer`.

**Update:** a second commit paces by the room the backend reports instead of a clock-based estimate. Under CPU load the estimate drifted and caused the very gaps this PR fixes (see Change).

### Why
On Linux, audio dropped out several times a second whenever the TUI was running: ~385 gaps per minute at 1 MS/s NFM in kitty, none headless. Listening was impractical.

### Cause
The prebuffer and depth cap rely on CoreAudio's client-side queue. PulseAudio and WASAPI size their device buffer from `blocksize` and block in `play()`; at 512 frames that is ~10 ms of headroom, so any stall of the audio thread longer than that (a garbage collection, the GIL held elsewhere) was an audible gap. The underflow counter also counted every block pushed, hiding the problem.

### Change
On non-CoreAudio backends (CoreAudio is unchanged):
- Request a 400 ms device buffer and start it full of silence. Its capacity is the room the backend reports at open (PipeWire grants ~60%).
- Pace by the room the backend reports free: PulseAudio's writable size (tracked client-side, no server round trip) or WASAPI's buffer size less padding. One `play()` per batch instead of one per 512-frame block (~10x fewer round trips), entered only once it fits, so `play()` never polls.
- Count an underflow when the device has nothing of ours left; trim a backlog beyond two buffers (net of the room the device has now) to half a buffer.
- A backend that reports no room falls back to blocking writes.

The first commit estimated the fill from wall-clock time and took any `play()` slower than 2 ms as "full". Under CPU load `play()` is often slow with room to spare, so the estimate read full while the device ran dry: uncounted 30-40 ms gaps, and backlog trims that discarded ~0.4 s every few seconds. The second commit removes the estimate.

### Testing
- New tests with a device fake that plays in real time and reports its room: granted capacity, pacing with 512- and 2048-frame quanta, a slow write with room, underflow, a stall of the worker vs. real backlog, the fallback, the PulseAudio reader. The slow-write and worker-stall tests fail on the first commit.
- Live (Linux, PipeWire, kitty, WFM on a PlutoSDR, 90 s): at 15 MS/s the first commit made 9 trims and ~100 gaps with no underflow counted; the second made no trims, and the few gaps left (the radio delivered ~1% under 15 MS/s) were all counted. At 10 MS/s: no gaps.
- Heavy CPU load (10 busy processes, synthetic WFM file, 60 s): first commit 14 trims, audio reached the device at 89-93% of real time; second commit 1 trim (the startup burst), 100%.
- On a real PipeWire sink the reported capacity is 245 ms, what the first commit measured by timing.
- Full suite on this branch: 1398 passed; pre-commit clean.
- Test machine: Intel Core i7-8700, Ubuntu 24.04, Python 3.14.8 (free-threaded).

### Notes
The WASAPI path follows the same logic but was not tested on Windows.

## #4: Kitty graphics: a shared-memory object per frame (fixes SIGBUS), cleaned up

[floens/tsdr#4](https://github.com/floens/tsdr/pull/4), opened 2026-10-11, branch `pr/kitty-shm-frames`.

### Why
Kitty itself could crash (SIGBUS in its image cache) while showing tsdr's spectrum or waterfall, taking the terminal window and everything in it down, or show one frame's pixels under another's header.

### Cause
On POSIX every frame recreated one fixed shared-memory name per image. When the terminal had not yet read a queued frame, the next frame replaced that object before it was sized or filled.

### Change
- Each frame gets its own object, `tsdr_<pid>_<image_id>_<seq>`, which the terminal unlinks after reading.
- Frames the terminal never reads are unlinked by tsdr: beyond 64 outstanding, when an image is removed or the widget unmounts, and at exit. Without this, unread frames stay in `/dev/shm` (RAM) for good; e.g. each test-suite run left ~500 objects (~150 MB).
- A frame the terminal unlinks while tsdr cleans it up is tolerated.

### Testing
New tests: queued frames keep distinct buffers; unread frames are unlinked past the cap and on unmount; the unlink race. A full test run now leaves nothing in `/dev/shm`. Full suite on this branch: 1389 passed; pre-commit clean.

## #5: TUI: keep garbage-collection pauses short

[floens/tsdr#5](https://github.com/floens/tsdr/pull/5), opened 2026-10-11, branch `pr/gc-pauses`.

### Why
At high sample rates the TUI lost samples every few seconds: audio stalled, decoders saw gaps, and a PlutoSDR at 10 MS/s delivered ~9.8 MS/s inside the TUI against the full 10.00 MS/s headless. Each loss was a garbage collection stopping every thread for ~110-130 ms, longer than the radio's driver buffers at that rate.

### Cause
The collector was freeing ~33k objects/s of cyclic garbage, almost all Textual render output: `Strip.divide` caches `[self]` and `Strip.crop_extend` caches `self` in their no-op cases, so nearly every rendered line is a reference cycle (Textual 8.2.8; unchanged on Textual's main). The tuner also re-ran the screen layout on every stats event (~16/s) with unchanged text.

### Change
- `tui/strip_cycles.py`: return the strip without caching it in those two cases (same result, no cycle). Imported by `app.py`, like the existing `tty.py` patch.
- `main()` freezes what imports created (~250k objects) before the app exists, so collections skip it (a full collection drops from 36 to 13 ms on the test machine); app objects stay collectable.
- The tuner's device line updates only when its text changes.

### Testing
- New tests: whole-strip operations leave no cycle (and the unpatched method does); mounting the app freezes nothing; unchanged stats don't repaint.
- Live (PlutoSDR, 10 MS/s NFM, 200x55 kitty window, 78 s): 23 collections of ~130 ms -> none; 9.80 -> 10.00 MS/s delivered.
- Full suite on this branch: 1392 passed; pre-commit clean.
- Test machine: Intel Core i7-8700 (3.2 GHz), Ubuntu 24.04, Python 3.14.8 (free-threaded), Textual 8.2.8.

### Notes
The Strip change works around a Textual issue that could also be reported upstream there; it is harmless if Textual fixes it.

## #6: SoapySDR: report driver capabilities, count overflows, warn on low throughput

[floens/tsdr#6](https://github.com/floens/tsdr/pull/6), opened 2026-10-11, branch `pr/soapy-capabilities`.

### Why
- Soapy radios were treated as unlimited: tsdr didn't know their frequency range, discrete sample rates (e.g. the Airspy's two fixed rates) or gain step, so tuning limits, rate snapping and completion didn't match the hardware.
- Samples could be lost with no sign of it anywhere: driver overflows were skipped silently, and a radio delivering well below its sample rate (a network PlutoSDR asked for more than its link carries, ~15 MS/s on gigabit Ethernet) just made the audio stutter without explanation.

### Change
- Query `getFrequencyRange`, `getSampleRateRange` (discrete rates when the driver reports them) and the gain step; missing queries fall back to the old defaults.
- Count driver overflows and log them in aggregate (`soapy_overflow`); name read errors with `errToStr`.
- Every 10 s the I/O worker compares samples read with the device's actual rate; under 97% it logs `device_throughput_low` and, at most once a minute, shows a device error suggesting a lower rate. The first window after a start or rate change is skipped.

### Testing
New tests with fake devices for each. Full suite on this branch: 1396 passed; pre-commit clean.

## #7: DSP: multistage decimation with channel selection; WSJT upper sideband

[floens/tsdr#7](https://github.com/floens/tsdr/pull/7), opened 2026-10-11, branch `pr/multistage-decimation`.

### Why
Tuning around produced phantom signals. Strong stations elsewhere in the capture folded onto whatever channel was tuned, so you heard (and decoders tried to decode) signals that are not on that frequency. On a PlutoSDR at 10 MS/s, NOAA weather at 162.55 MHz played on 162.6125 MHz. This is a serious bug, not a performance tradeoff: at 10 MS/s the old filtering rejected such signals by 0.4 dB, i.e. not at all, which makes higher sample rates unusable for listening or decoding. WSJT had a related flaw: a signal below the dial produced the same audio as one above it, so the decoder saw the wrong sideband's signals.

### Cause
- The audio demods, WFM, WSJT and most decoders decimated from the capture rate in one short FIR (e.g. 64 taps at 208:1 for NFM at 10 MS/s). Its transition band was ~600 kHz wide, so stations near multiples of the output rate folded straight onto the channel: worst-case rejection 26.5 dB at 1 MS/s, 9.1 dB at 2 MS/s, 0.4 dB at 10 MS/s.
- WSJT took the real part of the IQ, which discards the difference between positive and negative frequencies.

### Change
- `radio/dsp/decimator.py`: `MultistageDecimator` (Crochiere and Rabiner), a cascade planned for the fewest multiply-accumulates, each stage stopping what would fold into the final passband: at least 81 dB of rejection from 1 to 20 MS/s. Used by the shared demod front end, WFM, WSJT, FMChannelizer/DMR, ACARS, FSK, TETRA, RDS and recording.
- The cascade passes everything up to its output Nyquist, so WFM and FMChannelizer now also select the channel with an FIR at the decimated rate (WFM: a station 75 kHz from a 50 kHz channel is 93-104 dB down).
- WSJT keeps the upper sideband only (lower sideband > 50 dB down).
- The decimating FIR kernels now vectorize, and the always-on frequency shift computes sin/cos once per 256 samples (8-13x faster, error < 1e-6).

### Cost
Proper filtering takes more arithmetic than the filter that let everything through: the NFM front end does 6.9 multiply-accumulates per input sample at 10 MS/s (old: 0.31) and 10.7 at 2 MS/s (old: 1.6). That is trivial on a modern CPU: 0.94 ms per 50 ms of 10 MS/s input on a 2017 desktop i7-8700, about 2% of one core, and the vectorized kernels absorb most of the increase.

### Testing
- New tests: cascade rejection and streaming continuity, rejection of folded stations through each front end, WFM and DMR neighbor rejection, WFM bandwidth narrowing, WSJT sideband rejection, frequency-shift phase continuity. Decoder tests on recorded samples pass.
- On PlutoSDR IQ captured at 10 MS/s, NOAA came through on 162.6125 MHz only 0.8 dB below its own channel with the old front end; the new one measures that channel's true power within 0.1 dB. Live, WFM on a broadcast station plays with RDS.
- Full suite on this branch: 1411 passed; pre-commit clean.
- Test machine: Intel Core i7-8700 (3.2 GHz), Ubuntu 24.04, Python 3.14.8 (free-threaded), numba 0.67.0.

## #8: SoapySDR: read the stream on a native thread

[floens/tsdr#8](https://github.com/floens/tsdr/pull/8), opened 2026-10-13, branch `pr/soapy-native-reader`.

Builds on #6; only the last commit is new here.

### Why
Even with shorter collections, anything that stalls Python for longer than the radio's driver buffer loses samples, and on a PlutoSDR at 10 MS/s that buffer is only ~100 ms. Lost samples mean audio gaps, broken decodes and false throughput warnings. Reading on a thread that never enters the interpreter makes capture immune to garbage collections and GIL contention.

### Change
`devices/soapy_reader.py`: a numba `nogil` loop calls the SoapySDR C API (`SoapySDRDevice_readStream`, via the handles the SWIG bindings hold) into a 0.5 s ring; `read_samples` drains it.
- Fenced publishing for weakly ordered CPUs; overruns counted and logged.
- A retune marks the ring: reads stop at the mark and each batch is labeled with the frequency its samples were captured at (`BuffersCapture`); nothing is dropped. Batch timestamps account for the backlog.
- Rate changes restart the ring; a loop stuck in the driver is never freed under it.
- Compiled in `open()` before the stream starts (SoapyAirspy stops streaming for good if the first read is ~0.2 s late); falls back to Python reads if compilation fails or the bindings expose no C handles.

### Testing
- New tests drive the real loop against a ctypes fake driver: continuity across ring wraps, overruns, driver errors and timeouts, interrupt, stuck-reader handling, retune labeling.
- Live, PlutoSDR at 10 MS/s with forced ~105 ms collections every second: 9.62 MS/s through Python reads vs 10.00 native; in the TUI under forced collections 9.37 -> 10.00 MS/s and 28 -> 4 audio stalls; holding a tuning key (25 retunes/s) loses no samples. Airspy R2: 10.00 and 2.50 MS/s.
- Full suite on this branch: 1414 passed; pre-commit clean.
- Test machine: Intel Core i7-8700 (3.2 GHz), Ubuntu 24.04, Python 3.14.8 (free-threaded), numba 0.67.0.

### Notes
Significant design change but it brings significant benefits. A few hours of live testing including running it at 15
MS/s on a PlutoSDR hasn't turned up any issues.

## #9: Zero-IF radios: keep the LO beside the channel; track the Pluto's analog filter

[floens/tsdr#9](https://github.com/floens/tsdr/pull/9), opened 2026-10-13, branch `pr/zero-if`.

### Why
Many popular SDRs are direct-conversion (zero-IF) receivers: the HackRF, the ADALM-Pluto and its derivatives (Pluto+ and others), LimeSDR, USRP and bladeRF. Their capture center carries LO leakage, DC offset, 1/f noise and second-order intermod products from strong signals anywhere in the passband. tsdr's center tuning put the demodulated channel exactly on that spot, so every channel you listen to sits on top of the radio's worst noise and spurs: on a Pluto+ on UHF, the channel at the LO was 7.5-9.7 dB noisier than one 50 kHz away, enough to bury weak signals.

On the Pluto, a second problem: its analog anti-alias filter stays at 18 MHz whatever the sample rate, so strong signals far outside the displayed span reach the ADC. They overload it (raising the noise floor across the band) and alias into the span as phantom signals.

### Change
- LO offset: while a demodulator runs, the hardware center sits one channel width (at least 50 kHz) from the dial; FrequencyShiftStage already moves the channel to baseband, so demodulators and decoders are unchanged. `DeviceConfig.lo_offset` (auto/below/above/off; auto = below on zero-IF radios, detected from the Soapy driver key) and `lo_offset_hz`; `o`/`O` keys and `config --lo-offset`/`--lo-offset-size`; persisted per device; the stats panel shows the LO. Center mode keeps the view centered on the dial.
- Analog filter: `DeviceConfig.analog_filter` (auto/track/fixed/off), by default 0.8x the sample rate on drivers that leave the filter alone (the Pluto); set at open and after every rate change, restored on close, shown in the stats panel; `config --analog-filter`.

### Testing
- New tests for the tuning policy (offset, range-edge mirroring, fit checks, free mode, a fixed-width decoder such as ADS-B keeping its own width when the bandwidth setting changes), persistence, commands, and the filter policy and worker behavior.
- Live, Pluto+ at 2 MS/s with a signal generator (gain 40): before, a tone 2-6 MHz outside the span overloaded the ADC from -35 dBm and raised the noise floor 16 dB at -30 dBm; with the filter at 0.8x the rate, no overload up to -20 dBm (the test limit), and a tone aliasing into the span was 24 dB lower. The filter costs 1.2 dB at 40% of the rate from center, for signal and noise alike.
- Full suite on this branch: 1454 passed; pre-commit clean.

### Notes
Overlaps #2, #6 and #8 in `device_context.py`, `io_worker.py` and `soapy.py`; I'll rebase it once those are settled.

## #10: Spectrum: average the FFT bins under each column of a zoomed view

[floens/tsdr#10](https://github.com/floens/tsdr/pull/10), opened 2026-10-13, branch `pr/spectrum-projection`.

### Why
Zooming in by even one step makes the spectrum trace much noisier and lifts it by several dB, so the noise floor looks rougher and signals look stronger than they are. At 10 MS/s with a 32768-point FFT on a 1600-pixel-wide plot of noise, the full band shows 1.3 dB of jitter between neighboring columns, with the highest column 3 dB above the median; one zoom step shows 7.4 dB of jitter, with the highest column 9.5 dB above the median. The full-band view itself is unaffected.

### Cause
For the full band, `project_spectrum` averages the bins under each column (`decimate_spectrum`). For any other view it takes one bin per column. A zoomed view at a high sample rate still has many bins per column (14 at one zoom step in the example above), so the trace shows single-bin noise.

### Change
When the view has more bins than columns, each column averages the bins it covers, as the full-band path does. When it has fewer (zoomed far in), the bins are stretched as before. Columns outside the capture stay blank.

A LO offset PR (#9) I plan to offer crops the full view about the dial, so without this change the full band would look noisy whenever the offset is on.

### Cost
One addition per FFT bin in view per redraw, the same as the full-band path. On the test machine, 140 µs per redraw at 1600 columns, the same as the full-band path (134 µs); the single-bin pick took 45 µs.

### Testing
- New tests: each column is the mean of its bins, a slightly cropped full view is as smooth as the full band, and columns past the capture are blank. The first two fail on the old code.
- Synthetic noise at 10 MS/s, 32768-point FFT, 1600 columns: jitter between neighboring columns 7.4 dB before, 2.0 dB after at one zoom step (14 bins per column); 8.4 dB before, 3.1 dB after at three steps (6 bins per column). Fewer bins per column average less, so a zoomed view stays a little rougher than the full band.
- Full suite on this branch: 1388 passed; pre-commit clean.
- Test machine: Intel Core i7-8700, Ubuntu 24.04, Python 3.14.8 (free-threaded).

## #11: TUI: the console takes the keyboard after a panel click; config changes from the pipeline reach the UI

[floens/tsdr#11](https://github.com/floens/tsdr/pull/11), opened 2026-10-14, branch `pr/tui-fixes`.

### Why
Two UI bugs, both seen in everyday use:
- **Typing into the console was ignored.** After a click on a panel (Stats, Performance, a decoder's), `` ` `` opened the console but every key typed was dropped, until the console line was clicked or the app was quit with Ctrl+C. It looked like a frozen keyboard and came and went, depending on whether a panel had been clicked.
- **The UI could keep showing an old config.** When the pipeline worker changes a device's config (the automatic gain stage stepping the RF gain), a change within 50 ms of another logged `event_handler_failed ... RuntimeError('no running event loop')`, and the UI stayed on the earlier config until the next change.

### Cause
- `SectionPanel`, the base of the Stats, Performance and decoder panels, is a Textual `VerticalScroll`, which takes focus when clicked. Opening the console marks its input active but leaves the focus where it was, so keys went to the panel, and the app, seeing the console active, let them fall through. The other click-only lists (memories, directory, decoder output, the console history) already set `can_focus = False`.
- The engine publishes `ConfigChangedEvent` on the thread that made the change. `TextualEventAdapter` coalesces those events and, for one inside its 50 ms refresh interval, starts a Textual timer, which can only start on the app's event loop; on the pipeline worker it raised.

### Change
- `SectionPanel.can_focus = False` (panels scroll by mouse; the app owns the keyboard), and opening the console focuses its input, so nothing that takes focus later can do this again. Either change alone fixes the case.
- Off the app's loop, the adapter posts `ConfigChanged` at once (`post_message` is thread-safe); such changes come a few a second at most, and the UI re-reads the config when it handles them. Changes made on the app's loop keep the coalescing.

### Cost
None measurable: one focus call when the console opens; at most a few extra messages a second from worker threads.

### Testing
- New tests, each failing on master: `tests/tui/test_console_focus.py` clicks the Stats panel in a running app, opens the console and types `help` (the input stayed empty); `tests/tui/test_textual_adapter.py` publishes three config changes from a worker thread into a running app (it logged the error and delivered one of three).
- Full suite on this branch: 1388 passed, 7 skipped; pre-commit clean.
- Test machine: Intel Core i7-8700 (3.2 GHz), Ubuntu 24.04, Python 3.14.8 (free-threaded), Textual 8.2.8.
