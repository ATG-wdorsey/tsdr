"""IQ recordings as FLAC files, through the system's libFLAC (ctypes: no Python package).

A recording is a 2-channel FLAC file, I in the first channel and Q in the second, as
16-bit integers (full scale 32768). FLAC's sample rate field can't hold
most radio rates: up to 655,350 Hz it holds the true rate, above that the rate in kS/s
(16 MS/s is written as 16,000 Hz). The true rate is in the ``TSDR_SAMPLE_RATE`` tag,
beside ``TSDR_CENTER_FREQUENCY``, ``TSDR_GAIN`` and ``TSDR_DEVICE``.

libFLAC 1.3 or later: ``libflac12t64`` on Ubuntu 24.04 (``libflac8`` on 22.04), ``brew
install flac`` on macOS, ``libFLAC.dll`` from the FLAC release on Windows. Its calls run
without the GIL; the decoder's per-frame callback holds it briefly.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tsdr.core.sdr.config import FLAC_DEFAULT_LEVEL

# The largest rate written as itself (FLAC's limit before libFLAC 1.4, which takes 1,048,575).
MAX_STREAMINFO_RATE = 655_350
TAG_SAMPLE_RATE = "TSDR_SAMPLE_RATE"
TAG_CENTER_FREQUENCY = "TSDR_CENTER_FREQUENCY"
TAG_GAIN = "TSDR_GAIN"
TAG_DEVICE = "TSDR_DEVICE"

# Levels 3-8 use it anyway; at 0-2 (1152 by default) it costs nothing more to encode, saves a
# little and takes a fifth off decoding (a Python callback per frame): 16 MS/s at level 1,
# 26 % of an i7-8700 core to encode, 39 % to decode.
_BLOCKSIZE = 4096

_LIBRARY_NAMES = (
    "libFLAC.so.12",
    "libFLAC.so.14",
    "libFLAC.so.8",
    "libFLAC.so",
    "/opt/homebrew/lib/libFLAC.dylib",
    "/usr/local/lib/libFLAC.dylib",
    "libFLAC.dylib",
    "libFLAC.dll",
    "FLAC.dll",
)

_ENCODER_STATES = (
    "OK",
    "UNINITIALIZED",
    "OGG_ERROR",
    "VERIFY_DECODER_ERROR",
    "VERIFY_MISMATCH_IN_AUDIO_DATA",
    "CLIENT_ERROR",
    "IO_ERROR",
    "FRAMING_ERROR",
    "MEMORY_ALLOCATION_ERROR",
)
_ENCODER_INIT_STATUSES = (
    "OK",
    "ENCODER_ERROR",
    "UNSUPPORTED_CONTAINER",
    "INVALID_CALLBACKS",
    "INVALID_NUMBER_OF_CHANNELS",
    "INVALID_BITS_PER_SAMPLE",
    "INVALID_SAMPLE_RATE",
    "INVALID_BLOCK_SIZE",
    "INVALID_MAX_LPC_ORDER",
    "INVALID_QLP_COEFF_PRECISION",
    "BLOCK_SIZE_TOO_SMALL_FOR_LPC_ORDER",
    "NOT_STREAMABLE",
    "INVALID_METADATA",
    "ALREADY_INITIALIZED",
)
_DECODER_INIT_STATUSES = (
    "OK",
    "UNSUPPORTED_CONTAINER",
    "INVALID_CALLBACKS",
    "MEMORY_ALLOCATION_ERROR",
    "ERROR_OPENING_FILE",
    "ALREADY_INITIALIZED",
)
_DECODER_STATES = (
    "SEARCH_FOR_METADATA",
    "READ_METADATA",
    "SEARCH_FOR_FRAME_SYNC",
    "READ_FRAME",
    "END_OF_STREAM",
    "OGG_ERROR",
    "SEEK_ERROR",
    "ABORTED",
    "MEMORY_ALLOCATION_ERROR",
    "UNINITIALIZED",
)
_DECODER_END_OF_STREAM = 4
_DECODER_ERRORS = (
    "LOST_SYNC",
    "BAD_HEADER",
    "FRAME_CRC_MISMATCH",
    "UNPARSEABLE_STREAM",
    "BAD_METADATA",
)
# FLAC__FrameHeader: blocksize, sample_rate, channels, channel_assignment, bits_per_sample (32 bits each).
_FRAME_BLOCKSIZE, _FRAME_CHANNELS, _FRAME_BITS = 0, 8, 16
_METADATA_TYPE_VORBIS_COMMENT = 4
_WRITE_CONTINUE, _WRITE_ABORT = 0, 1


class FlacError(OSError):
    """libFLAC is missing, or refused a file."""


def _name(names: tuple[str, ...], value: int) -> str:
    return names[value] if 0 <= value < len(names) else str(value)


class _CommentEntry(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("entry", ctypes.c_char_p)]


_P = ctypes.c_void_p
_BOOL, _U32, _U64 = ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64
# Raw addresses: the frame (its header's first field is the block size) and the per-channel
# sample arrays, read without making ctypes pointer objects (a callback runs per frame).
_WRITE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _P, _P, _P, _P)
_ERROR_CALLBACK = ctypes.CFUNCTYPE(None, _P, ctypes.c_int, _P)

_SIGNATURES: dict[str, tuple[Any, list[Any]]] = {
    "FLAC__stream_encoder_new": (_P, []),
    "FLAC__stream_encoder_delete": (None, [_P]),
    "FLAC__stream_encoder_set_channels": (_BOOL, [_P, _U32]),
    "FLAC__stream_encoder_set_bits_per_sample": (_BOOL, [_P, _U32]),
    "FLAC__stream_encoder_set_sample_rate": (_BOOL, [_P, _U32]),
    "FLAC__stream_encoder_set_streamable_subset": (_BOOL, [_P, _BOOL]),
    "FLAC__stream_encoder_set_compression_level": (_BOOL, [_P, _U32]),
    "FLAC__stream_encoder_set_blocksize": (_BOOL, [_P, _U32]),
    "FLAC__stream_encoder_set_metadata": (_BOOL, [_P, ctypes.POINTER(_P), _U32]),
    "FLAC__stream_encoder_init_file": (ctypes.c_int, [_P, ctypes.c_char_p, _P, _P]),
    "FLAC__stream_encoder_process_interleaved": (_BOOL, [_P, ctypes.POINTER(ctypes.c_int32), _U32]),
    "FLAC__stream_encoder_finish": (_BOOL, [_P]),
    "FLAC__stream_encoder_get_state": (ctypes.c_int, [_P]),
    "FLAC__metadata_object_new": (_P, [ctypes.c_int]),
    "FLAC__metadata_object_delete": (None, [_P]),
    "FLAC__metadata_object_vorbiscomment_append_comment": (_BOOL, [_P, _CommentEntry, _BOOL]),
    "FLAC__stream_decoder_new": (_P, []),
    "FLAC__stream_decoder_delete": (None, [_P]),
    "FLAC__stream_decoder_init_file": (
        ctypes.c_int,
        [_P, ctypes.c_char_p, _WRITE_CALLBACK, _P, _ERROR_CALLBACK, _P],
    ),
    "FLAC__stream_decoder_process_single": (_BOOL, [_P]),
    "FLAC__stream_decoder_reset": (_BOOL, [_P]),
    "FLAC__stream_decoder_get_state": (ctypes.c_int, [_P]),
    "FLAC__stream_decoder_finish": (_BOOL, [_P]),
}
# Not in libFLAC's public header, but exported (the flac tool's --no-md5-sum): the MD5 of
# the audio costs a tenth of a core at 16 MS/s, for a check nothing here makes.
_OPTIONAL = {"FLAC__stream_encoder_set_do_md5": (_BOOL, [_P, _BOOL])}


@functools.cache
def _lib() -> ctypes.CDLL:
    found = ctypes.util.find_library("FLAC")
    names = ((found,) if found else ()) + _LIBRARY_NAMES
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            for fn, (restype, argtypes) in _SIGNATURES.items():
                getattr(lib, fn).restype = restype
                getattr(lib, fn).argtypes = argtypes
        except OSError, AttributeError:
            continue
        for fn, (restype, argtypes) in _OPTIONAL.items():
            if hasattr(lib, fn):
                getattr(lib, fn).restype = restype
                getattr(lib, fn).argtypes = argtypes
        return lib
    raise FlacError(
        "libFLAC not found: install it (Ubuntu: apt install libflac12t64, "
        "macOS: brew install flac) to record or play FLAC IQ files"
    )


def available() -> bool:
    try:
        _lib()
    except FlacError:
        return False
    return True


def _path_arg(path: Path) -> bytes:
    # libFLAC opens files itself: UTF-8 names on Windows, the file system's bytes elsewhere.
    return str(path).encode("utf-8") if os.name == "nt" else os.fsencode(path)


def streaminfo_rate(sample_rate: float) -> int:
    """The rate written in FLAC's own field: the true one when it fits, else in kS/s."""
    rate = round(sample_rate)
    if rate > MAX_STREAMINFO_RATE:
        rate = round(sample_rate / 1000)
    return max(1, rate)


def format_tag_number(value: float) -> str:
    """A rate or frequency for a tag: whole hertz without a fraction (16000000, 44100.5)."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class FlacInfo:
    """A FLAC file's stream header and tags (read without libFLAC)."""

    sample_rate: float  # the true rate (the TSDR_SAMPLE_RATE tag, else FLAC's field)
    channels: int
    bits_per_sample: int
    total_samples: int  # 0 = unknown (a file whose writer never finished)
    tags: dict[str, str] = field(default_factory=dict)


def read_info(path: str | Path) -> FlacInfo:
    """STREAMINFO and the Vorbis comments of a FLAC file. Raises ValueError if it isn't one."""
    with open(path, "rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError(f"{path}: not a FLAC file")
        streaminfo: bytes | None = None
        tags: dict[str, str] = {}
        last = False
        while not last:
            header = f.read(4)
            if len(header) < 4:
                break
            last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            if block_type == 0:
                streaminfo = f.read(length)
            elif block_type == _METADATA_TYPE_VORBIS_COMMENT:
                tags = _parse_comments(f.read(length))
            else:
                f.seek(length, os.SEEK_CUR)
    if streaminfo is None or len(streaminfo) < 18:
        raise ValueError(f"{path}: FLAC file without a stream header")
    packed = int.from_bytes(streaminfo[10:18], "big")
    rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    bits = ((packed >> 36) & 0x1F) + 1
    total = packed & 0xFFFFFFFFF
    true_rate = float(rate)
    if TAG_SAMPLE_RATE in tags:
        try:
            true_rate = float(tags[TAG_SAMPLE_RATE])
        except ValueError:
            pass
    return FlacInfo(true_rate, channels, bits, total, tags)


def _parse_comments(block: bytes) -> dict[str, str]:
    tags: dict[str, str] = {}
    try:
        (vendor_length,) = struct.unpack_from("<I", block, 0)
        pos = 4 + vendor_length
        (count,) = struct.unpack_from("<I", block, pos)
        pos += 4
        for _ in range(count):
            (length,) = struct.unpack_from("<I", block, pos)
            entry = block[pos + 4 : pos + 4 + length].decode("utf-8", "replace")
            pos += 4 + length
            name, sep, value = entry.partition("=")
            if sep:
                tags[name.upper()] = value
    except struct.error:
        pass
    return tags


class FlacWriter:
    """Writes one 2-channel FLAC file. One thread at a time; `finish()` completes it."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: float,
        bits_per_sample: int = 16,
        level: int = FLAC_DEFAULT_LEVEL,
        tags: dict[str, str] | None = None,
    ) -> None:
        lib = _lib()
        self._lib = lib
        self._path = Path(path)
        self._encoder = lib.FLAC__stream_encoder_new()
        if not self._encoder:
            raise MemoryError("libFLAC: no memory for an encoder")
        self._comments = None
        try:
            enc = self._encoder
            ok = (
                lib.FLAC__stream_encoder_set_channels(enc, 2)
                and lib.FLAC__stream_encoder_set_bits_per_sample(enc, bits_per_sample)
                and lib.FLAC__stream_encoder_set_sample_rate(enc, streaminfo_rate(sample_rate))
                # Most radio rates aren't in FLAC's streamable subset (a frame header can't name them).
                and lib.FLAC__stream_encoder_set_streamable_subset(enc, 0)
                and lib.FLAC__stream_encoder_set_compression_level(enc, level)
                and lib.FLAC__stream_encoder_set_blocksize(enc, _BLOCKSIZE)
            )
            if not ok:
                raise FlacError(f"libFLAC refused the settings for {self._path}")
            if hasattr(lib, "FLAC__stream_encoder_set_do_md5"):
                lib.FLAC__stream_encoder_set_do_md5(enc, 0)
            all_tags = {TAG_SAMPLE_RATE: format_tag_number(sample_rate), **(tags or {})}
            self._set_comments(all_tags)
            status = lib.FLAC__stream_encoder_init_file(enc, _path_arg(self._path), None, None)
            if status != 0:
                raise FlacError(
                    f"libFLAC can't write {self._path}: {_name(_ENCODER_INIT_STATUSES, status)}"
                )
        except BaseException:
            self._release()
            raise

    def _set_comments(self, tags: dict[str, str]) -> None:
        lib = self._lib
        self._comments = lib.FLAC__metadata_object_new(_METADATA_TYPE_VORBIS_COMMENT)
        if not self._comments:
            raise MemoryError("libFLAC: no memory for tags")
        for name, value in tags.items():
            text = f"{name}={value}".encode()
            entry = _CommentEntry(len(text), text)
            if not lib.FLAC__metadata_object_vorbiscomment_append_comment(self._comments, entry, 1):
                raise MemoryError("libFLAC: no memory for tags")
        blocks = (_P * 1)(self._comments)
        if not lib.FLAC__stream_encoder_set_metadata(self._encoder, blocks, 1):
            raise FlacError(f"libFLAC refused the tags for {self._path}")

    def write(self, interleaved: np.ndarray) -> None:
        """Encode samples given as int32 values, I and Q interleaved (C order)."""
        assert interleaved.dtype == np.int32 and interleaved.flags.c_contiguous
        n = len(interleaved) // 2
        if n == 0:
            return
        ptr = interleaved.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        if not self._lib.FLAC__stream_encoder_process_interleaved(self._encoder, ptr, n):
            raise FlacError(f"libFLAC failed writing {self._path}: {self._state()}")

    def finish(self) -> None:
        """Write the last frame and the final stream header, and close the file."""
        if not self._encoder:
            return
        try:
            if not self._lib.FLAC__stream_encoder_finish(self._encoder):
                raise FlacError(f"libFLAC failed finishing {self._path}: {self._state()}")
        finally:
            self._release()

    def _state(self) -> str:
        return _name(_ENCODER_STATES, self._lib.FLAC__stream_encoder_get_state(self._encoder))

    def _release(self) -> None:
        # Deleting an encoder finishes it first (closing its file) if it hasn't been.
        if self._encoder:
            encoder, self._encoder = self._encoder, None
            self._lib.FLAC__stream_encoder_delete(encoder)
        if self._comments:
            comments, self._comments = self._comments, None
            self._lib.FLAC__metadata_object_delete(comments)


class FlacReader:
    """Decodes a 2-channel FLAC IQ file in order: `read(n)` returns up to n samples as
    int32 I/Q pairs, shape (n, 2); an empty array at the end. `rewind()` starts over.
    A damaged file or a read error raises, on every read until a rewind. `close()` may
    come from another thread: it waits for a read under way."""

    def __init__(self, path: str | Path) -> None:
        self.info = read_info(path)
        if self.info.channels != 2:
            raise ValueError(f"{path}: {self.info.channels} channels, not an I/Q pair")
        lib = _lib()
        self._lib = lib
        self._path = Path(path)
        # Decoded samples not yet read: I in row 0, Q in row 1, columns [0, _filled).
        self._buf = np.empty((2, 0), dtype=np.int32)
        self._addresses = (0, 0)
        self._filled = 0
        self._error: str | None = None
        self._failed: BaseException | None = None
        self._lock = threading.Lock()
        # Kept for the decoder's lifetime: libFLAC calls them.
        self._write_cb = _WRITE_CALLBACK(self._on_frame)
        self._error_cb = _ERROR_CALLBACK(self._on_error)
        self._decoder = lib.FLAC__stream_decoder_new()
        if not self._decoder:
            raise MemoryError("libFLAC: no memory for a decoder")
        status = lib.FLAC__stream_decoder_init_file(
            self._decoder, _path_arg(self._path), self._write_cb, None, self._error_cb, None
        )
        if status != 0:
            self.close()
            raise FlacError(f"libFLAC can't read {path}: {_name(_DECODER_INIT_STATUSES, status)}")

    def _reserve(self, n: int) -> None:
        if self._filled + n <= self._buf.shape[1]:
            return
        grown = np.empty((2, max(2 * self._buf.shape[1], self._filled + n, 1 << 16)), np.int32)
        grown[:, : self._filled] = self._buf[:, : self._filled]
        self._buf = grown
        self._addresses = (grown[0].ctypes.data, grown[1].ctypes.data)

    def _on_frame(self, _decoder: Any, frame: int, buffer: int, _client: Any) -> int:
        try:
            n = ctypes.c_uint32.from_address(frame + _FRAME_BLOCKSIZE).value
            channels = ctypes.c_uint32.from_address(frame + _FRAME_CHANNELS).value
            bits = ctypes.c_uint32.from_address(frame + _FRAME_BITS).value
            if channels != 2 or bits != self.info.bits_per_sample:
                raise FlacError(
                    f"{self._path}: a frame of {channels} channels at {bits} bits in a stream "
                    f"of 2 at {self.info.bits_per_sample}"
                )
            i_samples, q_samples = (ctypes.c_void_p * 2).from_address(buffer)
            if not i_samples or not q_samples:
                raise FlacError(f"libFLAC gave a frame without samples in {self._path}")
            self._reserve(n)
            offset = 4 * self._filled
            ctypes.memmove(self._addresses[0] + offset, i_samples, 4 * n)
            ctypes.memmove(self._addresses[1] + offset, q_samples, 4 * n)
            self._filled += n
        except BaseException as e:  # noqa: BLE001 - raised again by the caller
            self._failed = e
            return _WRITE_ABORT
        return _WRITE_CONTINUE

    def _on_error(self, _decoder: Any, status: int, _client: Any) -> None:
        self._error = _name(_DECODER_ERRORS, status)

    def read(self, n: int) -> np.ndarray:
        with self._lock:
            decoder = self._open_decoder()
            lib = self._lib
            # libFLAC's process_single returns false once, at an abort or a read error, and
            # true (doing nothing) after it: any state past END_OF_STREAM ends the reading.
            while self._filled < n and self._failed is None:
                state = lib.FLAC__stream_decoder_get_state(decoder)
                if state == _DECODER_END_OF_STREAM:
                    break
                if state > _DECODER_END_OF_STREAM or not lib.FLAC__stream_decoder_process_single(
                    decoder
                ):
                    state = lib.FLAC__stream_decoder_get_state(decoder)
                    if self._failed is None:
                        self._failed = FlacError(
                            f"libFLAC stopped reading {self._path}: {_name(_DECODER_STATES, state)}"
                        )
            if self._failed is not None:
                raise self._failed
            return self._take(n)

    def _open_decoder(self) -> Any:
        if not self._decoder:
            raise FlacError(f"{self._path} is closed")
        return self._decoder

    def _take(self, n: int) -> np.ndarray:
        n = min(n, self._filled)
        taken = np.ascontiguousarray(self._buf[:, :n].T)
        rest = self._filled - n
        if rest:
            self._buf[:, :rest] = self._buf[:, n : self._filled]
        self._filled = rest
        return taken

    @property
    def decode_error(self) -> str | None:
        """The last damage libFLAC reported and skipped (LOST_SYNC at a cut-off end, say)."""
        return self._error

    def rewind(self) -> None:
        with self._lock:
            decoder = self._open_decoder()
            self._filled, self._failed, self._error = 0, None, None
            if not self._lib.FLAC__stream_decoder_reset(decoder):
                raise FlacError(f"libFLAC can't rewind {self._path}")

    def close(self) -> None:
        with self._lock:
            if self._decoder:
                decoder, self._decoder = self._decoder, None
                self._lib.FLAC__stream_decoder_finish(decoder)
                self._lib.FLAC__stream_decoder_delete(decoder)

    def __enter__(self) -> FlacReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def to_complex64(pairs: np.ndarray, bits_per_sample: int) -> np.ndarray:
    """I/Q integer pairs, shape (n, 2), as complex64 (full scale 1.0)."""
    scale = np.float32(1.0 / (1 << (bits_per_sample - 1)))
    out = np.empty(len(pairs), dtype=np.complex64)
    out.real = pairs[:, 0] * scale
    out.imag = pairs[:, 1] * scale
    return out


def load(path: str | Path) -> tuple[np.ndarray, FlacInfo]:
    """A whole FLAC IQ file as complex64, with its header and tags (decoded in parts into
    one array: the file's size in memory once, not twice)."""
    with FlacReader(path) as reader:
        bits = reader.info.bits_per_sample
        out = np.empty(reader.info.total_samples, dtype=np.complex64)
        filled = 0
        while len(pairs := reader.read(1 << 20)):
            if filled + len(pairs) > len(out):  # a file cut short holds no sample count
                grown = np.empty(max(2 * len(out), filled + len(pairs)), dtype=np.complex64)
                grown[:filled] = out[:filled]
                out = grown
            out[filled : filled + len(pairs)] = to_complex64(pairs, bits)
            filled += len(pairs)
        return out[:filled], reader.info
