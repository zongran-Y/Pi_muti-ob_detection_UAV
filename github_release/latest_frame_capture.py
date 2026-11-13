"""Drain a live camera continuously and expose a single replaceable latest frame."""

import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class FramePacket:
    frame: object
    sequence: int
    read_started_at: float
    received_at: float


class LatestFrameCapture:
    def __init__(self, capture):
        self.capture = capture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._last_delivered = 0
        self._received = self._delivered = self._overwritten = 0
        self._ended = False
        self._error = None

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Camera reader cannot be started twice")
        self._thread = threading.Thread(target=self._run, name="latest-camera-frame", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                ok, frame = self.capture.read()
                received = time.perf_counter()
                if not ok or self._stop.is_set():
                    break
                with self._condition:
                    if self._latest is not None and self._latest.sequence > self._last_delivered:
                        self._overwritten += 1
                    self._received += 1
                    self._latest = FramePacket(frame, self._received, started, received)
                    self._condition.notify_all()
        except Exception as exc:
            self._error = exc
        finally:
                                                                                   
            try:
                self.capture.release()
            except Exception as exc:
                if self._error is None:
                    self._error = exc
            with self._condition:
                self._ended = True
                self._condition.notify_all()

    def read(self, timeout=2.0):
        if self._thread is None:
            raise RuntimeError("Start camera reader before reading")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Camera frame timeout must be finite and positive")
        deadline = time.perf_counter() + timeout
        with self._condition:
            while self._latest is None or self._latest.sequence <= self._last_delivered:
                if self._error is not None:
                    raise RuntimeError("Camera read failed: %s" % self._error) from self._error
                if self._ended or self._stop.is_set():
                    return False, None
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("No fresh camera frame within %.2f seconds" % timeout)
                self._condition.wait(remaining)
            packet = self._latest
            self._last_delivered = packet.sequence
            self._delivered += 1
            return True, packet

    def stats(self):
        with self._condition:
            return {
                "frames_received": self._received,
                "frames_delivered": self._delivered,
                "frames_overwritten": self._overwritten,
            }

    def close(self, timeout=2.0):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is None:
            self.capture.release()
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()
