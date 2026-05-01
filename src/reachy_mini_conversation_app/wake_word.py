"""Wake-word gate that pauses mic forwarding until a hotword is detected."""

from __future__ import annotations

import time
import logging
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample


logger = logging.getLogger(__name__)


WAKE_WORD_SAMPLE_RATE = 16000
WAKE_WORD_CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz


class WakeWordGate:
    """Gate mic frames behind a wake-word detector.

    States:
      - ASLEEP: frames are not forwarded; each frame is fed to the wake-word
        detector. When the hotword is detected, the gate transitions to AWAKE.
      - AWAKE: frames are forwarded to the realtime handler. After
        ``sleep_timeout_s`` of inactivity, the gate transitions back to ASLEEP.

    Activity is bumped externally (from the realtime handler's speech events
    or from the wake-word detection itself).
    """

    def __init__(
        self,
        wakeword_model: str,
        sleep_timeout_s: float = 30.0,
        detection_threshold: float = 0.5,
        vad_threshold: float = 0.0,
        rms_floor: float = 100.0,
        on_wake: Optional[Callable[[], None]] = None,
        on_sleep: Optional[Callable[[], None]] = None,
    ):
        """Load the openWakeWord model and configure thresholds.

        ``rms_floor`` is the cheapest pre-filter: any 80 ms chunk whose RMS
        (on int16 scale) is below this value is treated as silence and skips
        the openWakeWord ONNX inference entirely. Set to 0 to disable.

        ``vad_threshold`` enables openWakeWord's built-in Silero VAD; off by
        default because it also runs ONNX, which makes the RMS pre-filter
        the more effective lever on low-power hardware.
        """
        from openwakeword.model import Model
        import openwakeword.utils as oww_utils

        try:
            oww_utils.download_models([wakeword_model])
        except Exception as e:
            logger.debug("openwakeword.download_models warning: %s", e)

        model_kwargs: dict = {"wakeword_models": [wakeword_model]}
        if vad_threshold and vad_threshold > 0:
            model_kwargs["vad_threshold"] = vad_threshold
        self._model = Model(**model_kwargs)
        self._wakeword_name = next(iter(self._model.models.keys()))
        self._threshold = detection_threshold
        self._sleep_timeout_s = sleep_timeout_s
        self._rms_floor = float(rms_floor)
        self._stats_chunks_skipped = 0

        self._is_awake = False
        self._last_activity = time.monotonic()
        self._buffer: NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._on_wake = on_wake
        self._on_sleep = on_sleep

        # Debug stats while asleep
        self._first_frame_logged = False
        self._stats_window_start = time.monotonic()
        self._stats_peak_score = 0.0
        self._stats_peak_rms = 0.0
        self._stats_chunks_processed = 0

        logger.info(
            "WakeWordGate ready: wakeword=%s, sleep_timeout=%.1fs, threshold=%.2f",
            self._wakeword_name,
            sleep_timeout_s,
            detection_threshold,
        )

    @property
    def wakeword_name(self) -> str:
        """Name of the active wake-word model."""
        return self._wakeword_name

    def is_awake(self) -> bool:
        """Return whether the gate is currently letting frames through.

        Auto-sleeps after ``sleep_timeout_s`` of inactivity.
        """
        if self._is_awake and (time.monotonic() - self._last_activity) > self._sleep_timeout_s:
            self.sleep(reason="inactivity timeout")
        return self._is_awake

    def wake(self, *, reason: str = "manual") -> None:
        """Force the gate awake."""
        was_asleep = not self._is_awake
        if was_asleep:
            logger.info("WakeWordGate: AWAKE (%s)", reason)
        self._is_awake = True
        self._last_activity = time.monotonic()
        self._buffer = np.zeros(0, dtype=np.int16)
        if was_asleep and self._on_wake is not None:
            try:
                self._on_wake()
            except Exception as e:
                logger.warning("on_wake callback failed: %s", e)

    def sleep(self, *, reason: str = "manual") -> None:
        """Force the gate asleep."""
        was_awake = self._is_awake
        if was_awake:
            logger.info("WakeWordGate: ASLEEP (%s)", reason)
        self._is_awake = False
        self._buffer = np.zeros(0, dtype=np.int16)
        # NOTE: do NOT call self._model.reset() here. It clears openWakeWord's
        # internal audio embedding buffer; on the next sleep -> wake cycle the
        # model needs several seconds to refill it and detection scores stay
        # near zero in the meantime.
        if was_awake and self._on_sleep is not None:
            try:
                self._on_sleep()
            except Exception as e:
                logger.warning("on_sleep callback failed: %s", e)

    def notify_activity(self) -> None:
        """Bump the inactivity timer (call on user/assistant speech events)."""
        self._last_activity = time.monotonic()

    def should_forward(self, sample_rate: int, audio_frame: NDArray[np.int16]) -> bool:
        """Process a mic frame; return True if it should be forwarded to the handler."""
        if self.is_awake():
            return True

        try:
            if not self._first_frame_logged:
                logger.info(
                    "WakeWordGate first frame: sample_rate=%d, shape=%s, dtype=%s, abs_max=%g",
                    sample_rate,
                    audio_frame.shape,
                    audio_frame.dtype,
                    float(np.abs(audio_frame).max()) if audio_frame.size else 0.0,
                )
                self._first_frame_logged = True

            mono = self._to_mono_int16(audio_frame)
            if sample_rate != WAKE_WORD_SAMPLE_RATE:
                target = int(len(mono) * WAKE_WORD_SAMPLE_RATE / sample_rate)
                if target <= 0:
                    return False
                mono_f = resample(mono.astype(np.float32), target)
                mono = np.clip(mono_f, -32768, 32767).astype(np.int16)

            self._buffer = np.concatenate([self._buffer, mono])

            while len(self._buffer) >= WAKE_WORD_CHUNK_SAMPLES:
                chunk = self._buffer[:WAKE_WORD_CHUNK_SAMPLES]
                self._buffer = self._buffer[WAKE_WORD_CHUNK_SAMPLES:]
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                self._stats_peak_rms = max(self._stats_peak_rms, rms)
                self._stats_chunks_processed += 1

                if self._rms_floor > 0 and rms < self._rms_floor:
                    self._stats_chunks_skipped += 1
                    continue

                scores = self._model.predict(chunk)
                score = float(scores.get(self._wakeword_name, 0.0))
                self._stats_peak_score = max(self._stats_peak_score, score)
                if score >= self._threshold:
                    self.wake(reason=f"wakeword '{self._wakeword_name}' detected (score={score:.2f})")
                    return True

            now = time.monotonic()
            if now - self._stats_window_start >= 5.0:
                logger.info(
                    "WakeWordGate (asleep) last 5s: chunks=%d skipped=%d rms_peak=%.0f score_peak=%.3f",
                    self._stats_chunks_processed,
                    self._stats_chunks_skipped,
                    self._stats_peak_rms,
                    self._stats_peak_score,
                )
                self._stats_window_start = now
                self._stats_peak_score = 0.0
                self._stats_peak_rms = 0.0
                self._stats_chunks_processed = 0
                self._stats_chunks_skipped = 0
        except Exception as e:
            logger.warning("WakeWordGate: detection failed (%s); forwarding to be safe", e)
            self.wake(reason="detector error")
            return True

        return False

    @staticmethod
    def _to_mono_int16(audio: NDArray) -> NDArray[np.int16]:
        """Coerce audio (any shape, int or float) to a 1-D int16 mono array."""
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            audio = audio[:, 0]
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        return audio
