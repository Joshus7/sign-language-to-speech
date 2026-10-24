"""Offline speech: pyttsx3 running in its own thread so speaking never blocks the video."""
from __future__ import annotations

import logging
import queue
import threading

log = logging.getLogger(__name__)


class Speaker:
    """Queue short phrases for speech. Old phrases are dropped if speech falls behind,
    so it never reads out something from long ago."""

    MAX_QUEUE = 2

    def __init__(self, enabled: bool = True, rate: int = 175, volume: float = 1.0):
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._ready = threading.Event()
        self.available = False
        self._thread.start()
        self._ready.wait(5)

    def say(self, text: str | None) -> None:
        if not text or not self.enabled or not self.available:
            return
        while self._q.qsize() >= self.MAX_QUEUE:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(text)

    def set_rate(self, rate: int) -> None:
        self.rate = rate

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            self.available = True
        except Exception as exc:  # no speech driver (e.g. headless CI)
            log.warning("Voice disabled: %s", exc)
            self._ready.set()
            return
        self._ready.set()
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                log.warning("Speech failed: %s", exc)
        try:
            engine.stop()
        except Exception:
            pass
