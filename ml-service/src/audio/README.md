# Audio Preprocessing & Spectrogram Extraction Module

Smart Horizon 2026 — SH-DST-02 — Team ID: SHIH26-TID-320

## Overview

The `audio` module provides Log-Mel Spectrogram extraction and waveform normalization utilities for PyTorch tensors.

## Components

- **`spectrogram.py` (`LogMelSpectrogramExtractor`)**: Converts 1D raw audio waveforms (16 kHz) into standardized Log-Mel Spectrogram representations `(B, 80, T)`.
