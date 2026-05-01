"""Standalone wake-word debug: captures mic and prints openWakeWord scores live.

Usage:
    python scripts/debug_wake_word.py [--model hey_jarvis] [--device <id>]

Press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
import time
import argparse

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz


def list_devices() -> None:
    """Print available input devices."""
    print("Available input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']} — {dev['max_input_channels']} ch{mark}")


def main() -> None:
    """Capture mic and stream scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hey_jarvis", help="openWakeWord model name")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--list-devices", action="store_true", help="List devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    print(f"Loading wake-word model: {args.model}")
    model = Model(wakeword_models=[args.model])
    wakeword_name = next(iter(model.models.keys()))
    print(f"Loaded model: {wakeword_name}")
    print(f"Threshold: {args.threshold}")
    print(f"Sample rate: {SAMPLE_RATE} Hz, chunk: {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES * 1000 // SAMPLE_RATE} ms)")
    if args.device is None:
        print(f"Using default input device: {sd.query_devices(sd.default.device[0])['name']}")
    else:
        print(f"Using input device #{args.device}: {sd.query_devices(args.device)['name']}")
    print()
    print("Speak the wake word. Showing live RMS + score (peak per second). Ctrl+C to stop.\n")

    buffer = np.zeros(0, dtype=np.int16)
    last_print = time.monotonic()
    peak_score = 0.0
    peak_rms = 0.0

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio callback: extend buffer with the latest mic frame."""
        nonlocal buffer
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        # indata is float32 in [-1, 1]; convert to int16
        mono = indata[:, 0] if indata.ndim == 2 else indata
        int16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16)
        buffer = np.concatenate([buffer, int16])

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=args.device,
            callback=callback,
        ):
            while True:
                while len(buffer) >= CHUNK_SAMPLES:
                    chunk = buffer[:CHUNK_SAMPLES]
                    buffer = buffer[CHUNK_SAMPLES:]
                    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                    scores = model.predict(chunk)
                    score = float(scores.get(wakeword_name, 0.0))
                    peak_score = max(peak_score, score)
                    peak_rms = max(peak_rms, rms)

                    if score >= args.threshold:
                        print(f"\n*** DETECTED *** {wakeword_name} score={score:.3f} rms={rms:.0f}")

                now = time.monotonic()
                if now - last_print >= 1.0:
                    bar_len = max(0, min(40, int(peak_score * 40)))
                    bar = "#" * bar_len + "-" * (40 - bar_len)
                    print(f"  rms_peak={peak_rms:6.0f}  score_peak={peak_score:.3f}  [{bar}]")
                    peak_score = 0.0
                    peak_rms = 0.0
                    last_print = now

                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
