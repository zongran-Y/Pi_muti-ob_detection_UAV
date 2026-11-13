"""One detector owner, one replaceable pending image, one latest result."""
import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionPacket:
    frame: object
    sequence: int
    received_at: float
    completed_at: float
    detections: list
    tracks: object
    inference_ms: float
    context: dict
    started_at: object = None

    @property
    def stream_epoch(self):
        """Camera/model stream generation captured when this packet was submitted."""
        return self.context.get("stream_epoch", self.context.get("epoch"))

    @property
    def active_epoch(self):
        """Active-target generation for target-bound work; full-frame YOLO is reusable."""
        return self.context.get("active_epoch")

    @property
    def source_timestamp(self):
        return self.received_at


class LatestDetectionWorker:
    def __init__(self, detect, with_context=False):
        self.detect = detect
        self.with_context = bool(with_context)
        self._condition = threading.Condition()
        self._thread = None
        self._stopped = False
        self._busy = False
        self._pending = self._result = self._error = None
        self._submitted = self._completed = self._pending_overwritten = self._results_overwritten = 0
        self._processing_ms = self._queue_wait_ms = self._inference_ms = 0.0

    def start(self):
        with self._condition:
            if self._thread is not None or self._stopped:
                raise RuntimeError('Detector worker cannot be started twice')
            self._thread = threading.Thread(target=self._run, name='latest-yolo-frame', daemon=True)
            self._thread.start()
        return self

    def submit(self, frame, sequence, received_at, context):
        if not math.isfinite(received_at):
            raise ValueError('Detection frame time must be finite')
                                                                      
        image = frame.copy()
        with self._condition:
            if self._stopped or self._thread is None:
                raise RuntimeError('Detector worker is not running')
            if self._error is not None:
                raise RuntimeError('Detector failed: %s' % self._error) from self._error
            self._pending_overwritten += self._pending is not None
            self._pending = (image, sequence, received_at, dict(context))
            self._submitted += 1
            self._condition.notify_all()

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                    if self._stopped:
                        break
                    image, sequence, received_at, context = self._pending
                    self._pending = None
                    self._busy = True
                started_at = time.perf_counter()
                detections, tracks, inference_ms = (self.detect(image, context)
                    if self.with_context else self.detect(image))
                completed_at = time.perf_counter()
                result = DetectionPacket(image, sequence, received_at, completed_at,
                                         detections, tracks, inference_ms, context, started_at)
                with self._condition:
                    self._completed += 1
                    self._processing_ms += (completed_at - started_at) * 1000
                    self._queue_wait_ms += max(0.0, (started_at - received_at) * 1000)
                    self._inference_ms += inference_ms
                    self._results_overwritten += self._result is not None
                    self._result = result
                    self._busy = False
        except Exception as error:
            with self._condition:
                self._error = error
                self._busy = False
        finally:
            with self._condition:
                self._condition.notify_all()

    def poll(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError('Detector failed: %s' % self._error) from self._error
            result, self._result = self._result, None
            return result

    def stats(self):
        with self._condition:
            return dict(submitted=self._submitted, completed=self._completed,
                        pending_overwritten=self._pending_overwritten,
                        results_overwritten=self._results_overwritten,
                        busy=self._busy, pending=self._pending is not None,
                        mean_processing_ms=self._processing_ms / self._completed if self._completed else None,
                        mean_queue_wait_ms=self._queue_wait_ms / self._completed if self._completed else None,
                        mean_inference_ms=self._inference_ms / self._completed if self._completed else None,
                        error=str(self._error) if self._error is not None else None)

    def close(self, timeout=2.0):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        return self._thread is None or not self._thread.is_alive()
