# Acknowledgments

TSDR includes code derived from or inspired by the following projects, and a model trained in part on the data
credited at the end:

## DAB/DAB+ Decoder

- **welle.io** by Albrecht Lohofener et al.
  https://github.com/AlbrechtL/welle.io
- **qt-dab** by Jan van Katwijk
  https://github.com/JvanKatwijk/qt-dab

## ADS-B Decoder

- **dump1090-fa** by FlightAware / Oliver Jowett
  https://github.com/flightaware/dump1090
- **"The 1090 Megahertz Riddle"** by Junzi Sun
  https://mode-s.org/decode/

## DMR Decoder

- **dsdcc** by f4exb
  https://github.com/f4exb/dsdcc
- **SDRangel** by Edouard Griffiths (f4exb)
  https://github.com/f4exb/sdrangel

## AMBE+2 Vocoder

- **mbelib** by mbelib contributors
  https://github.com/szechyjs/mbelib

## TETRA Decoder

- **osmo-tetra** by Osmocom
  https://github.com/osmocom/osmo-tetra

## TETRA ACELP Vocoder

- **ETSI EN 300 395-2 V1.3.1** reference implementation

## RDS Decoder

- **PySDR** by Dr. Marc Lichtman
  https://pysdr.org

## CW Morse Decoder

- **fldigi** by W1HKJ et al.
  https://github.com/w1hkj/fldigi

## FT8 / FT4 (WSJT-X family) Decoder

- **ft8_lib** by Karlis Goba (YL3JG)
  https://github.com/kgoba/ft8_lib
- **WSJT-X** by Joe Taylor (K1JT) et al.
  https://wsjt.sourceforge.io

## APRS (AFSK1200 / AX.25) Decoder

- **Dire Wolf** by John Langner (WB2OSZ)
  https://github.com/wb2osz/direwolf
- **multimon-ng** by Elias Oenal et al.
  https://github.com/EliasOenal/multimon-ng

## ACARS Decoder

- **acarsdec** by Thierry Leconte (TLeconte)
  https://github.com/TLeconte/acarsdec
- **acars-message-documentation** by airframes.io (label reference)
  https://github.com/airframesio/acars-message-documentation

## Voice Detector (VSQ) Training Data

- **MUSAN: A Music, Speech, and Noise Corpus** by David Snyder, Guoguo Chen and Daniel Povey (2015),
  https://www.openslr.org/17/, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Its noise recordings were mixed into synthetic training examples without speech (open and quiet
  microphones, key-ups) for the voice detector's model (`src/tsdr/radio/voice/voice.npz`). No MUSAN
  audio is included in TSDR.
