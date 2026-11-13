"""YOLO acquisition, frame-to-frame visual tracking, and bounded prediction.

Tracking adapters consume xyxy boxes. OpenCV/NumPy are only needed when a real
adapter is constructed; the lock state machine is independently testable.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass
from roi_affine_tracker import ROIAffineTracker as _FlowTracker


class _KCFTracker:
    def __init__(self, constructor):
        self.tracker = constructor()

    def init(self, frame, bbox):
        x1, y1, x2, y2 = (round(value) for value in bbox)
        result = self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        return result is None or bool(result)                               

    def update(self, frame):
        ok, rect = self.tracker.update(frame)
        self.last_failure_reason = '' if ok else 'kcf_lost'
        if not ok:
            return False, None
        x, y, width, height = rect
        return True, (x, y, x + width, y + height)


class _LegacyFlowTracker:
    method = 'legacy_lk_translation'
    def __init__(self, cv2):
        import numpy as np
        self.cv2, self.np = cv2, np

    def _features(self, gray, bbox):
        mask = self.np.zeros(gray.shape, dtype=self.np.uint8)
        x1, y1, x2, y2 = (round(value) for value in bbox)
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
        return self.cv2.goodFeaturesToTrack(
            gray, maxCorners=80, qualityLevel=0.01, minDistance=3,
            mask=mask, blockSize=3,
        )

    def init(self, frame, bbox):
        self.gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        self.points = self._features(self.gray, bbox)
        self.bbox = bbox
        self.frames = 0
        self.good_point_count = len(self.points) if self.points is not None else 0
        self.last_failure_reason = '' if self.good_point_count >= 4 else 'insufficient_features'
        return self.points is not None and len(self.points) >= 4

    def _failed(self, reason):
        self.last_failure_reason = reason
        return False, None

    def update(self, frame):
        cv2, np = self.cv2, self.np
        self.good_point_count = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        settings = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))
        points, forward, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **settings)
        if points is None or forward is None or not np.isfinite(points).all():
            return self._failed('forward_flow_failed')
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, points, None, **settings)
        if backward is None or back_status is None:
            return self._failed('backward_flow_failed')
        error = np.max(np.abs(backward - self.points).reshape(-1, 2), axis=1)
        good = (forward.reshape(-1) != 0) & (back_status.reshape(-1) != 0) & (error < 1.5)
        self.good_point_count = int(good.sum())
        if self.good_point_count < max(4, math.ceil(len(self.points) * 0.5)):
            return self._failed('insufficient_consistent_points')
        old = self.points.reshape(-1, 2)[good]
        new = points.reshape(-1, 2)[good]
        dx, dy = np.median(new - old, axis=0)
        x1, y1, x2, y2 = self.bbox
        bbox = (float(x1 + dx), float(y1 + dy), float(x2 + dx), float(y2 + dy))
        height, width = gray.shape
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
            return self._failed('roi_out_of_view')
        inside = ((new[:, 0] >= bbox[0]) & (new[:, 0] <= bbox[2])
                  & (new[:, 1] >= bbox[1]) & (new[:, 1] <= bbox[3]))
        self.points = new[inside].reshape(-1, 1, 2).astype(np.float32)
        if len(self.points) < 4:
            return self._failed('points_left_roi')
        self.gray, self.bbox = gray, bbox
        self.frames += 1
        if len(self.points) < 20 or self.frames % 10 == 0:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            bx1, by1, bx2, by2 = (round(v) for v in bbox)
            mask[by1:by2, bx1:bx2] = 255
            for point in self.points.reshape(-1, 2):
                cv2.circle(mask, tuple(round(float(v)) for v in point), 4, 0, -1)
            extra = cv2.goodFeaturesToTrack(gray, maxCorners=max(1, 80 - len(self.points)),
                                           qualityLevel=.01, minDistance=3, mask=mask, blockSize=3)
            if extra is not None and len(self.points) < 80:
                self.points = np.concatenate((self.points, extra[:80 - len(self.points)]))
        self.last_failure_reason = ''
        return True, bbox


def create_tracker_factory(cv2, preference='auto', flow_config=None):
    if preference in ('auto', 'kcf'):
        for namespace in (cv2, getattr(cv2, 'legacy', None)):
            constructor = getattr(namespace, 'TrackerKCF_create', None)
            if callable(constructor):
                return lambda: _KCFTracker(constructor), 'kcf'
        if preference == 'kcf':
            raise ValueError('This OpenCV build has no KCF; use --visual-tracker flow or a contrib-enabled build')
    available = all(callable(getattr(cv2, name, None)) for name in (
        'cvtColor', 'goodFeaturesToTrack', 'calcOpticalFlowPyrLK',
    ))
    if available:
        if preference == 'flow-legacy':
            factory = lambda: _LegacyFlowTracker(cv2)
            factory.method = 'legacy_lk_translation'
            return factory, 'flow-legacy'
        if callable(getattr(cv2, 'estimateAffinePartial2D', None)) and callable(getattr(cv2, 'fillConvexPoly', None)):
            factory = lambda: _FlowTracker(cv2, flow_config)
            factory.method = 'roi_lk_ransac'
            return factory, 'flow'
        if preference == 'auto':
            factory = lambda: _LegacyFlowTracker(cv2)
            factory.method = 'legacy_lk_translation'
            return factory, 'flow-legacy'
        raise ValueError('ROI flow needs OpenCV estimateAffinePartial2D; use flow-legacy for comparison')
    if preference in ('flow', 'flow-legacy'):
        raise ValueError('This OpenCV build has no sparse optical-flow tracking support')
    return None, 'unavailable'


@dataclass(frozen=True)
class LockedTarget:
    bbox: tuple
    confidence: float
    class_id: int
    track_id: int
    position_source: str
    valid_until: object = None
    motion_reset: bool = False


@dataclass
class LockResult:
    observation: object
    raw_detections: list
    tracks: object
    inference_ms: object
    detector_ran: bool
    status: str
    detector_track_id: object = None
    position_age_s: object = None
    detection_age_s: object = None
    detector_accepted: bool = False
    detector_result_age_ms: object = None
    association_reason: str = 'not_run'
    association_candidates: object = None
    recovered: bool = False
    position_alignment: str = 'none'
    visual_update_ok: bool = False
    recovery_pending_count: int = 0
    visual_failure_reason: str = ''
    visual_point_count: object = None
    visual_tracking_ms: float = 0.
    visual_init_ms: float = 0.
    alignment_ms: float = 0.
    history_store_ms: float = 0.
    visual_inlier_count: object = None
    visual_inlier_ratio: object = None
    visual_reprojection_error: object = None
    visual_fb_error: object = None
    visual_scale: object = None
    visual_rotation_deg: object = None
    visual_roi_area: object = None
    alignment_steps: int = 0
    alignment_reason: str = ''
    alignment_method: str = ''
    alignment_budget_exceeded: bool = False
    lk_fwd_ok: object = None
    lk_bwd_ok: object = None
    lk_roundtrip_ok: object = None
    lk_visual_used: bool = False
    lk_ransac_reject_reason: str = 'not_run'
    detector_reanchor_pending_count: int = 0
    detector_reanchored: bool = False


class TargetLock:
    def __init__(self, tracker_factory, interval=0.25, prediction_timeout=0.25,
                 release_timeout=1.0, gate=0.35, max_frame_gap=0.5,
                 appearance=None, projector=None, recovery_timeout=3.0,
                 appearance_threshold=0.78, recovery_confirmations=2, max_detection_age=0.35,
                 single_target_recovery=False, use_sort=False, recovery_gate=None,
                 recovery_confirm_window=.75, recovery_area_ratio=2., detector_reanchor=True,
                 detector_reanchor_after=.75, detector_reanchor_confidence=.5):
        values = (interval, prediction_timeout, release_timeout, gate, max_frame_gap)
        if not all(math.isfinite(value) for value in values) or min(interval, gate, max_frame_gap) <= 0:
            raise ValueError('Tracking intervals and gate must be finite and positive')
        if not 0 <= prediction_timeout < release_timeout:
            raise ValueError('Require 0 <= prediction timeout < lock release timeout')
        if (not all(math.isfinite(v) for v in (recovery_timeout, appearance_threshold, max_detection_age))
                or recovery_timeout < release_timeout or not 0 < appearance_threshold <= 1
                or recovery_confirmations < 2 or max_detection_age <= 0):
            raise ValueError('Invalid identity recovery or detection age configuration')
        self.factory = tracker_factory
        self.interval, self.prediction_timeout = interval, prediction_timeout
        self.release_timeout, self.gate, self.max_frame_gap = release_timeout, gate, max_frame_gap
        self.counter = self.detector_calls = self.visual_frames = self.prediction_frames = 0
        self.detector_hit_frames = self.confirmed_detection_frames = 0
        self.backend_errors = 0
        self.last_error = None
        self.shape = self.last_frame_at = self.last_detector_at = None
        self.appearance, self.projector = appearance, projector
        self.recovery_timeout, self.appearance_threshold = recovery_timeout, appearance_threshold
        self.recovery_confirmations, self.max_detection_age = recovery_confirmations, max_detection_age
        self.recovery_gate = gate if recovery_gate is None else recovery_gate
        if (not all(math.isfinite(v) for v in (self.recovery_gate, recovery_confirm_window, recovery_area_ratio))
                or not gate <= self.recovery_gate <= max(gate, 1.) or recovery_confirm_window <= 0
                or not 1 <= recovery_area_ratio <= 4):
            raise ValueError('Invalid recovery search gate, confirmation window or area ratio')
        self.recovery_confirm_window, self.recovery_area_ratio = recovery_confirm_window, recovery_area_ratio
        if (not all(math.isfinite(v) for v in (detector_reanchor_after,detector_reanchor_confidence))
                or detector_reanchor_after <= 0 or not 0 <= detector_reanchor_confidence <= 1):
            raise ValueError('Invalid detector reanchor age/confidence')
        self.detector_reanchor = detector_reanchor
        self.detector_reanchor_after, self.detector_reanchor_confidence = detector_reanchor_after, detector_reanchor_confidence
        self.pending_reanchor = None
        self.detector_reanchors = 0
        self.memory = self.pending_recovery = None
        self.active_id = None
        self.epoch = 0
        self.last_requested_at = None
        self.identity_recoveries = self.stale_results = self.rejected_results = self.tracker_starts = 0
        self.visual_failures = 0
        self.needs_recovery = False
        self.appearance_conflicts = 0
        self.tracker_init_failures = 0
        self.last_init_ms = 0.
        self.single_target_recovery = single_target_recovery
        self.reason_counts = Counter()
        self.last_detection_result = None
        self.motion = None
        self._sort_ids = set()
        if use_sort:
            from sort.continuous import ContinuousSortTrack
            self.motion = ContinuousSortTrack(recovery_timeout)
        self._release()

    def _release(self, forget=False):
        self.pending_reanchor = None
        self.target = self.tracker = self.box = None
        self.active_id = None
        self.last_measured_at = self.last_confirmed_at = None
        self.velocity = (0.0, 0.0)
        self.velocity_samples = 0
        retain_identity = self.memory is not None and (self.memory['gallery'] or self.single_target_recovery)
        if forget or not retain_identity:
            self.pending_recovery = None
        self.needs_recovery = bool(retain_identity)
        if forget:
            self.memory = None
            self.needs_recovery = False
            self.epoch += 1
            if self.motion is not None:
                self.motion.reset()
        elif self.memory is not None and not retain_identity:
            self.memory = None
        if not retain_identity and self.motion is not None:
            self.motion.reset()

    def _box(self, bbox, width, height):
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            return None
        x1, y1, x2, y2 = bbox
        clipped = (max(0.0, float(x1)), max(0.0, float(y1)),
                   min(float(width), float(x2)), min(float(height), float(y2)))
        return clipped if clipped[2] - clipped[0] >= 2 and clipped[3] - clipped[1] >= 2 else None

    def _distance(self, first, second, width, height):
        return math.hypot(((first[0] + first[2]) - (second[0] + second[2])) / width,
                          ((first[1] + first[3]) - (second[1] + second[3])) / height)

    def _overlap(self, first, second):
        overlap = (max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
                   * max(0.0, min(first[3], second[3]) - max(first[1], second[1])))
        area = ((first[2] - first[0]) * (first[3] - first[1])
                + (second[2] - second[0]) * (second[3] - second[1]) - overlap)
        return overlap / area if area > 0 else 0.0

    @staticmethod
    def _similar_shape(first, second):
        return all(.8 <= (first[end] - first[start]) / (second[end] - second[start]) <= 1.25
                   for start, end in ((0, 2), (1, 3)))

    def _forecast(self, timestamp, width, height):
        if self.motion is not None:
            forecast = self.motion.peek(timestamp)
            return self._box(forecast, width, height) if forecast is not None else None
        age = max(0.0, timestamp - self.last_measured_at)
        dt = min(age, self.prediction_timeout)
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        return self._box((self.box[0] + dx, self.box[1] + dy,
                          self.box[2] + dx, self.box[3] + dy), width, height)

    def _measure(self, bbox, timestamp, restart_motion=False):
        if self.motion is not None:
            self.active_id = self.motion.correct(bbox, timestamp, restart_motion=restart_motion)
            if self.active_id not in self._sort_ids:
                self._sort_ids.add(self.active_id)
                self.counter += 1
                self.epoch += 1
                self.memory = None
        if restart_motion:
            self.velocity = (0.0, 0.0)
            self.velocity_samples = 0
        if not restart_motion and self.last_measured_at is not None and timestamp > self.last_measured_at:
            dt = timestamp - self.last_measured_at
            rate = (((bbox[0] + bbox[2]) - (self.box[0] + self.box[2])) / (2 * dt),
                    ((bbox[1] + bbox[3]) - (self.box[1] + self.box[3])) / (2 * dt))
            alpha = 1.0 if self.velocity_samples == 0 else 0.5
            self.velocity = tuple(alpha * value + (1 - alpha) * previous
                                  for value, previous in zip(rate, self.velocity))
            self.velocity_samples += 1
        self.box, self.last_measured_at = bbox, timestamp

    def _start_tracker(self, frame, bbox):
        started = time.perf_counter()
        try:
            self.tracker_starts += 1
            tracker = self.factory()
            self.tracker = tracker if tracker.init(frame, bbox) else None
        except Exception as error:
            self.backend_errors += 1
            self.last_error = str(error)
            self.tracker = None
        if self.tracker is None:
            self.needs_recovery = True
            self.tracker_init_failures += 1
        self.last_init_ms = (time.perf_counter()-started)*1000

    def detection_due(self, timestamp):
        return (self.target is None or self.tracker is None or self.last_requested_at is None
                or timestamp - self.last_requested_at >= self.interval)

    def mark_detection_requested(self, timestamp):
        self.last_requested_at = timestamp

    def detection_context(self, timestamp):
        reference = (self.motion.recovery_reference(timestamp, self.prediction_timeout)
                     if self.motion is not None else self.box)
        last_reference = self.box or (self.memory['bbox'] if self.memory else None)
        if reference is None and self.memory is not None:
            reference = self.memory['bbox']
        return dict(epoch=self.epoch, reference=reference,
                    last_reference=last_reference,
                    track_id=self.active_id or (self.memory['id'] if self.memory else None))

    def _describe(self, frame, bbox):
        return self.appearance.describe(frame, bbox) if self.appearance is not None else None

    def _similarity(self, descriptor):
        if descriptor is None or self.memory is None or not self.memory['gallery']:
            return None
        return max(self.appearance.similarity(item, descriptor) for item in self.memory['gallery'])

    def _remember(self, timestamp, descriptor=None):
        if self.memory is None:
            self.memory = dict(id=self.active_id, class_id=self.target.class_id, gallery=[])
        self.memory.update(bbox=self.box, timestamp=timestamp)
        if descriptor is not None:
            gallery = self.memory['gallery']
            if not gallery or self._similarity(descriptor) >= self.appearance_threshold:
                gallery.append(descriptor)
                if len(gallery) > 4:
                    del gallery[1]

    def _select(self, detections, frame, reference, width, height, last_reference=None, timestamp=None, allow_reanchor=False):
        diagnostics, matches = [], []
        valid = [item for item in detections if self._box(item.bbox, width, height) is not None]
        if self.target is None and self.memory is None:
            reason = 'acquired' if len(valid) == 1 else ('ambiguous' if len(valid) > 1 else 'no_detection')
            return (valid[0] if len(valid) == 1 else None), False, reason, diagnostics
        expected_class = self.target.class_id if self.target is not None else self.memory['class_id']
        if self.single_target_recovery and sum(item.class_id == expected_class for item in valid) > 1:
            return None, False, 'ambiguous', diagnostics
        reference = reference or self.box or self.memory['bbox']
        measured_at = self.last_measured_at if self.last_measured_at is not None else self.memory['timestamp']
        recovery_age = max(0., (timestamp if timestamp is not None else self.last_frame_at)-measured_at)
        growth = min(1., max(0., (recovery_age-self.prediction_timeout)
                             / (self.release_timeout-self.prediction_timeout)))
        recovery_gate = self.gate + (self.recovery_gate-self.gate)*growth
        detector_support_age = max(0., (timestamp if timestamp is not None else self.last_frame_at)-self.last_confirmed_at) if self.last_confirmed_at is not None else 0.
        for item in valid:
            descriptor = self._describe(frame, item.bbox)
            similarity = self._similarity(descriptor)
            distance = self._distance(item.bbox, reference, width, height)
            last_distance = self._distance(item.bbox, last_reference, width, height) if last_reference else None
            overlap = self._overlap(item.bbox, reference)
            size_reference = last_reference if self.single_target_recovery and last_reference else reference
            ref_area = (size_reference[2] - size_reference[0]) * (size_reference[3] - size_reference[1])
            area_ratio = (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]) / ref_area
            spatial = distance <= self.gate and overlap >= .1
            recovery = (self.target is None or not spatial
                        or ((self.appearance is not None or self.single_target_recovery) and self.needs_recovery))
            if item.class_id != expected_class:
                reason = 'class_mismatch'
            elif not .25 <= area_ratio <= 4:
                reason = 'size_gate'
            elif recovery:
                if self.single_target_recovery:
                    recovery_distance = min(distance, last_distance) if last_distance is not None else distance
                    if recovery_distance > recovery_gate:
                        reason = 'recovery_motion_gate'
                    elif not 1/self.recovery_area_ratio <= area_ratio <= self.recovery_area_ratio:
                        reason = 'recovery_size_gate'
                    else:
                        reason = 'recovery_candidate'
                elif distance > max(self.gate, .65):
                    reason = 'recovery_distance_gate'
                elif similarity is None:
                    reason = 'appearance_unavailable'
                elif similarity < self.appearance_threshold:
                    reason = 'appearance_mismatch'
                else:
                    reason = 'recovery_candidate'
            elif not self.single_target_recovery and similarity is not None and similarity < .55:
                reason = 'appearance_mismatch'
            else:
                reason = 'spatial_match'
            if self.appearance is None and recovery and not self.single_target_recovery:
                reason = 'spatial_gate'
                                                                            
                                                                                 
                                                                                   
            reanchor = (reason not in ('spatial_match','recovery_candidate')
                        and allow_reanchor and self.detector_reanchor and self.single_target_recovery
                        and self.target is not None and item.class_id == expected_class
                        and item.confidence >= self.detector_reanchor_confidence
                        and detector_support_age >= self.detector_reanchor_after)
            if reanchor:
                reason = 'detector_reanchor_candidate'
            diagnostics.append(dict(bbox=list(item.bbox), confidence=item.confidence, class_id=item.class_id,
                                    iou=overlap, center_distance=distance,
                                    last_center_distance=last_distance, area_ratio=area_ratio,
                                    reference_bbox=list(reference),
                                    last_reference_bbox=list(last_reference) if last_reference else None,
                                    recovery_gate=recovery_gate, recovery_age_s=recovery_age,
                                    recovery_area_ratio=self.recovery_area_ratio,
                                    detector_support_age_s=detector_support_age,
                                    appearance_score=similarity, reason=reason))
            if reason in ('spatial_match', 'recovery_candidate', 'detector_reanchor_candidate'):
                matches.append((item, False if reanchor else recovery, reanchor))
        if len(matches) != 1:
            return None, False, 'ambiguous' if len(matches) > 1 else ('association_rejected' if valid else 'no_detection'), diagnostics
        return matches[0][0], matches[0][1], 'detector_reanchor_candidate' if matches[0][2] else 'matched', diagnostics

    def _confirm_reanchor(self, bbox, timestamp, source_at, width, height):
        previous = self.pending_reanchor
        if previous is not None and source_at <= previous[3]:
            return False                                                           
        count = 1
        if previous is not None and 0 < timestamp-previous[1] <= self.recovery_confirm_window:
            previous_area = (previous[0][2]-previous[0][0])*(previous[0][3]-previous[0][1])
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            if self._distance(bbox,previous[0],width,height) <= self.gate and .5 <= area/previous_area <= 2:
                count = previous[2]+1
        self.pending_reanchor = (bbox,timestamp,count,source_at)
        return count >= self.recovery_confirmations

    def _confirm_recovery(self, bbox, timestamp, width, height):
        previous = self.pending_recovery
        count = 1
        if (previous is not None and 0 < timestamp - previous[1] <= self.recovery_confirm_window
                and self._distance(bbox, previous[0], width, height) <= self.gate):
            count = previous[2] + 1
        self.pending_recovery = (bbox, timestamp, count)
        return count >= self.recovery_confirmations

    def update(self, frame, timestamp, detect=None, *, detection=None):
        self.last_init_ms = 0.
        visual_tracking_ms = alignment_ms = history_store_ms = 0.
        visual_quality, alignment_quality = {}, {}
        height, width = frame.shape[:2]
        if not math.isfinite(timestamp) or min(width, height) <= 0:
            raise ValueError('Invalid tracking frame or timestamp')
        shape = (width, height)
        if self.last_frame_at is not None:
            if shape != self.shape or timestamp <= self.last_frame_at:
                self._release(forget=True)
            elif timestamp - self.last_frame_at > self.max_frame_gap:
                self._release()
        if self.last_measured_at is not None and timestamp - self.last_measured_at > self.release_timeout:
            self._release()
        if self.memory is not None and timestamp - self.memory['timestamp'] > self.recovery_timeout:
            self.memory = self.pending_recovery = None
        if self.pending_recovery is not None and timestamp - self.pending_recovery[1] > self.recovery_confirm_window:
            self.pending_recovery = None
        if self.pending_reanchor is not None and timestamp-self.pending_reanchor[1] > self.recovery_confirm_window:
            self.pending_reanchor = None
        self.shape, self.last_frame_at = shape, timestamp
        if detect is None and callable(getattr(self.projector, 'observe', None)):
            started = time.perf_counter()
            self.projector.observe(frame, timestamp)
            history_store_ms = (time.perf_counter()-started)*1000
        visual = None
        visual_failure_reason, visual_point_count = '', None
        if self.tracker is not None:
            started = time.perf_counter()
            try:
                ok, bbox = self.tracker.update(frame)
                visual_quality = dict(getattr(self.tracker, 'diagnostics', {}))
                visual_point_count = getattr(self.tracker, 'good_point_count', None)
                visual = self._box(bbox, width, height) if ok else None
                if visual is not None and self._distance(visual, self.box, width, height) > self.gate:
                    visual = None
                    visual_failure_reason = 'visual_jump_gate'
                if visual is None:
                    visual_failure_reason = visual_failure_reason or getattr(self.tracker, 'last_failure_reason', 'visual_update_failed')
                    self.visual_failures += 1
                    self.needs_recovery = True
                    self.tracker = None
            except Exception as error:
                self.backend_errors += 1
                self.last_error = str(error)
                visual_failure_reason = 'backend_exception'
                self.visual_failures += 1
                self.needs_recovery = True
                self.tracker = None
            finally:
                visual_tracking_ms = (time.perf_counter()-started)*1000
        run_detector = detection is not None or (detect is not None and self.detection_due(timestamp))
        if visual is None:
            self.pending_reanchor = None
        detections, tracks, inference_ms = [], [], None
        chosen = None
        recovering = recovered = accepted = False
        reanchoring = reanchored = False
        result_age_ms = None
        alignment, reason, diagnostics = 'none', 'not_run', []
        source_frame, detected_at = frame, timestamp
        last_reference = self.box or (self.memory['bbox'] if self.memory else None)
        reference = visual or (self._forecast(timestamp, width, height)
                               if self.box is not None or self.motion is not None else None)
        if visual is None and self.motion is not None:
            reference = self.motion.recovery_reference(timestamp, self.prediction_timeout)
        if run_detector:
            if detection is not None:
                detections, tracks, inference_ms = detection.detections, detection.tracks, detection.inference_ms
                source_frame, detected_at = detection.frame, detection.received_at
                reference = detection.context.get('reference')
                last_reference = detection.context.get('last_reference')
                result_age_ms = (timestamp - detected_at) * 1000
                if (not math.isfinite(detected_at) or not 0 <= timestamp - detected_at <= self.max_detection_age
                        or source_frame.shape[:2] != frame.shape[:2]):
                    reason = 'stale_result'
                    self.stale_results += 1
                elif detection.context.get('epoch') != self.epoch:
                    reason = 'obsolete_lock_result'
                else:
                    chosen, recovering, reason, diagnostics = self._select(detections, source_frame, reference, width, height, last_reference, detected_at, allow_reanchor=visual is not None)
            else:
                self.mark_detection_requested(timestamp)
                detections, tracks, inference_ms = detect()
                chosen, recovering, reason, diagnostics = self._select(detections, frame, reference, width, height, last_reference, timestamp, allow_reanchor=visual is not None)
            reanchoring = reason == 'detector_reanchor_candidate'
            self.last_detector_at = detected_at
            self.detector_calls += 1
            self.detector_hit_frames += bool(detections)
                                                                              
                                                                          
        if (visual is not None and self.appearance is not None and not self.single_target_recovery
                and any(item['reason'] == 'appearance_mismatch' and item['iou'] >= .5
                        and item['center_distance'] <= self.gate
                        and self.target is not None and item['class_id'] == self.target.class_id
                        for item in diagnostics)):
            current_score = self._similarity(self._describe(frame, visual))
            if current_score is not None and current_score < .55:
                self.tracker, visual = None, None
                self.needs_recovery = True
                self.visual_failures += 1
                self.appearance_conflicts += 1
                reason = 'visual_appearance_conflict'
        current_box = None
        if chosen is not None:
            if (detection is not None and visual is not None and not recovering and not reanchoring
                    and reference is not None and self._overlap(chosen.bbox, reference) >= .5
                    and self._similar_shape(chosen.bbox, reference)):
                current_box, alignment = visual, 'visual_confirmation'
            elif detection is not None and timestamp > detected_at:
                started = time.perf_counter()
                try:
                    if callable(getattr(self.projector, 'project_at', None)):
                        current_box = self.projector.project_at(source_frame, chosen.bbox, frame, detected_at, timestamp)
                    else:
                        current_box = self.projector(source_frame, chosen.bbox, frame) if self.projector is not None else None
                except Exception as error:
                    self.backend_errors += 1
                    self.last_error = str(error)
                    current_box = None
                finally:
                    alignment_ms = (time.perf_counter()-started)*1000
                    alignment_quality = dict(getattr(self.projector, 'diagnostics', {}))
                current_box = self._box(current_box, width, height) if current_box is not None else None
                alignment = 'flow_aligned' if current_box is not None else 'alignment_failed'
                if current_box is None:
                    chosen, reason = None, 'alignment_failed'
            else:
                current_box, alignment = self._box(chosen.bbox, width, height), 'current_frame'
        if chosen is not None and reanchoring:
            if self._confirm_reanchor(current_box,timestamp,detected_at,width,height):
                reanchored = True
            else:
                chosen, reason = None, 'detector_reanchor_pending'
        elif run_detector and reason not in ('no_detection','stale_result','obsolete_lock_result','alignment_failed'):
            self.pending_reanchor = None
        if chosen is not None and recovering:
            if not self._confirm_recovery(current_box, timestamp, width, height):
                chosen, reason = None, 'recovery_pending'
            else:
                recovered = True
        elif chosen is None and run_detector and reason not in ('no_detection', 'stale_result', 'obsolete_lock_result', 'alignment_failed'):
            self.pending_recovery = None
        source, observation = 'none', None
        if chosen is not None:
            accepted = True
            self.confirmed_detection_frames += 1
            if self.target is None and self.motion is None:
                if self.memory is not None and recovering:
                    self.active_id = self.memory['id']
                else:
                    self.counter += 1
                    self.active_id = self.counter
                    self.memory = None
                    self.epoch += 1
            self.target = chosen
            self.needs_recovery = False
            self._measure(current_box, timestamp, restart_motion=recovered or reanchored)
            self.last_confirmed_at = detected_at
            self._remember(timestamp, self._describe(source_frame, chosen.bbox))
                                                                           
                                                                               
            if self.tracker is None or visual is None or tuple(current_box) != tuple(visual):
                self._start_tracker(frame, self.box)
            source = 'visual' if alignment == 'visual_confirmation' else 'detector'
            if source == 'visual':
                self.visual_frames += 1
            if reanchored:
                self.detector_reanchors += 1
                self.epoch += 1                                                     
                reason = 'detector_reanchored'
            elif recovered:
                self.identity_recoveries += 1
                reason = 'identity_recovered'
            else:
                reason = 'accepted'
            self.pending_recovery = None
            self.pending_reanchor = None
        elif visual is not None:
            self._measure(visual, timestamp)
            self._remember(timestamp)
            self.visual_frames += 1
            source = 'visual'
        elif self.target is not None and timestamp - self.last_measured_at <= self.prediction_timeout:
            forecast = self._forecast(timestamp, width, height)
            if forecast is not None:
                source = 'prediction'
                self.prediction_frames += 1
        if self.motion is not None and source not in ('detector', 'visual'):
            self.motion.advance(timestamp)
        if source != 'none':
            bbox = forecast if source == 'prediction' else self.box
            observation = LockedTarget(bbox, self.target.confidence if source == 'detector' else 0.0,
                                       self.target.class_id, self.active_id, source,
                                       self.last_measured_at + self.prediction_timeout if source == 'prediction'
                                       else timestamp + self.max_frame_gap, motion_reset=reanchored)
        if run_detector and detections and not accepted:
            self.rejected_results += 1
        if run_detector:
            self.reason_counts[reason] += 1
                                                                             
                                                                          
            self.last_detection_result = dict(
                source_timestamp=detected_at, consumed_timestamp=timestamp,
                target_count=len(detections), accepted=accepted,
                association_reason=reason, position_alignment=alignment,
                alignment_reason=alignment_quality.get('reason', ''),
                result_age_at_consumption_ms=result_age_ms,
            )
        status = ('predicting' if source == 'prediction' else 'locked_' + source) if observation else (
            'lost_hold' if self.target is not None else ('ambiguous' if len(detections) > 1 else 'searching'))
        return LockResult(
            observation, detections, tracks, inference_ms, run_detector, status,
            chosen.track_id if chosen is not None else None,
            timestamp - self.last_measured_at if self.last_measured_at is not None else None,
            timestamp - self.last_confirmed_at if self.last_confirmed_at is not None else None,
            accepted, result_age_ms, reason, diagnostics, recovered, alignment, visual is not None,
            self.pending_recovery[2] if self.pending_recovery is not None else 0,
            visual_failure_reason, visual_point_count,
            visual_tracking_ms=visual_tracking_ms, visual_init_ms=self.last_init_ms,
            alignment_ms=alignment_ms, history_store_ms=history_store_ms,
            visual_inlier_count=visual_quality.get('inlier_count'),
            visual_inlier_ratio=visual_quality.get('inlier_ratio'),
            visual_reprojection_error=visual_quality.get('reprojection_error'),
            visual_fb_error=visual_quality.get('fb_error'), visual_scale=visual_quality.get('scale'),
            visual_rotation_deg=visual_quality.get('rotation_deg'), visual_roi_area=visual_quality.get('roi_area'),
            lk_fwd_ok=visual_quality.get('fwd_ok'),
            lk_bwd_ok=visual_quality.get('bwd_ok'),
            lk_roundtrip_ok=visual_quality.get('roundtrip_ok'),
            lk_visual_used=visual is not None,
            lk_ransac_reject_reason=visual_quality.get('ransac_reject_reason', 'not_run'),
            alignment_steps=alignment_quality.get('steps', 0), alignment_reason=alignment_quality.get('reason', ''),
            alignment_method=alignment_quality.get('method', ''),
            alignment_budget_exceeded=alignment_quality.get('budget_exceeded', False),
            detector_reanchor_pending_count=self.pending_reanchor[2] if self.pending_reanchor else 0,
            detector_reanchored=reanchored,
        )

    def summary(self):
        return dict(lock_sessions=self.counter, detector_calls=self.detector_calls,
                    flow_method=getattr(self.factory, 'method', 'external_or_kcf'),
                    detector_hit_frames=self.detector_hit_frames,
                    confirmed_detection_frames=self.confirmed_detection_frames,
                    visual_frames=self.visual_frames, prediction_frames=self.prediction_frames,
                    tracker_ready=self.tracker is not None, backend_errors=self.backend_errors,
                    last_backend_error=self.last_error, identity_recoveries=self.identity_recoveries,
                    stale_results=self.stale_results, rejected_results=self.rejected_results,
                    tracker_starts=self.tracker_starts, visual_failures=self.visual_failures,
                    appearance_conflicts=self.appearance_conflicts,
                    tracker_init_failures=self.tracker_init_failures,
                    association_reasons=dict(self.reason_counts),
                    last_detection_result=dict(self.last_detection_result) if self.last_detection_result else None,
                    recovery_search_gate=self.recovery_gate,
                    recovery_confirm_window_s=self.recovery_confirm_window,
                    recovery_max_area_ratio=self.recovery_area_ratio,
                    detector_reanchors=self.detector_reanchors,
                    detector_reanchor_pending_count=self.pending_reanchor[2] if self.pending_reanchor else 0,
                    detector_reanchor_enabled=self.detector_reanchor and self.single_target_recovery,
                    detector_reanchor_after_s=self.detector_reanchor_after,
                    detector_reanchor_confidence=self.detector_reanchor_confidence,
                    identity_policy='single_target_continuity' if self.single_target_recovery else (
                        'appearance' if self.appearance is not None else 'disabled'),
                    motion_backend='timed_sort_single' if self.motion is not None else 'pixel_velocity')
