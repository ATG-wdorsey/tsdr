# IMBE 7200x4400 test vectors

- `bmh_gasline.imbe`: speech as IMBE frames, 11 bytes per frame holding the 88-bit parameter vector
  (u0..u7, most significant bit first), before the frame's error correction. The speech is the
  AMBE+2 recording in `../ambe/bmh_gasline.amb` (from
  [pbarfuss/mbelib-testing](https://github.com/pbarfuss/mbelib-testing), mbelib, ISC license), decoded
  to 8 kHz with mbelib and coded again with OP25's IMBE encoder (boatbod/op25 71abcd0,
  `lib/imbe_vocoder`, built as a local tool; only its output is here).
- `bmh_gasline_mbelib.npz`: the parameters mbelib decodes from each frame
  ([szechyjs/mbelib](https://github.com/szechyjs/mbelib) 9a04ed5, `imbe7200x4400.c`, the state advanced as
  `mbe_processImbe7200x4400Framef` does), as they stand before spectral enhancement: `w0` (radians per
  sample), `L`, `repeat`, `vl` (`Vl[1..L]`) and `ml` (`Ml[1..L]`).
