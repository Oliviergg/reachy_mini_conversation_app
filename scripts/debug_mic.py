"""Tiny mic check: capture 3s and print signal stats. No wake-word, no model."""

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
DURATION_S = 3


def main() -> None:
    """Record DURATION_S of audio and print signal stats."""
    print(f"Default input: {sd.query_devices(sd.default.device[0])['name']}")
    print(f"Recording {DURATION_S}s @ {SAMPLE_RATE} Hz... speak now")
    audio = sd.rec(int(SAMPLE_RATE * DURATION_S), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    a = audio.flatten()
    print(f"  samples : {len(a)}")
    print(f"  abs max : {float(np.abs(a).max()):.4f}")
    print(f"  abs mean: {float(np.abs(a).mean()):.4f}")
    print(f"  rms     : {float(np.sqrt(np.mean(a ** 2))):.4f}")
    if float(np.abs(a).max()) < 1e-5:
        print()
        print("==> Signal is ZERO. The mic is muted or the app lacks Microphone permission.")
        print("    macOS: System Settings -> Privacy & Security -> Microphone -> enable for Terminal/iTerm.")
        print("    Then fully quit the terminal app and reopen.")


if __name__ == "__main__":
    main()
