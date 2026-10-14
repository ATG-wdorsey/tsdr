# AMBE+2 2450 test vectors

- `bmh_gasline.amb`: a speech recording as AMBE+2 2450 frames (`.amb`: the bytes `.amb`, then per frame
  one byte of channel errors and 7 bytes holding the 49 bits), from
  [pbarfuss/mbelib-testing](https://github.com/pbarfuss/mbelib-testing) (mbelib, ISC license).
- `bmh_gasline_mbelib.npz`: the parameters mbelib decodes from each frame (after decoding, before spectral
  enhancement): `bad` (0 = speech, 2 = erasure, 3 = tone), `w0` (radians per sample), `L`, `voiced` (the
  number of voiced harmonics) and `ml` (`Ml[1..L]`). Made with
  [szechyjs/mbelib](https://github.com/szechyjs/mbelib) 9a04ed5 (`mbe_decodeAmbe2450Parms`, the state advanced
  as `mbe_processAmbe2450Dataf` does). The mbelib-testing variant interpolates w0 instead of the
  specification's table (up to 0.2 % apart); tsdr uses the table, as mbelib does.
