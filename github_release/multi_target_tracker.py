"""Bounded, source-time multi-target tracking for the UAV camera pipeline.

Each track owns a 6D constant-velocity Kalman state
``[cx, cy, log(w), log(h), vx, vy]``. YOLO updates position and size; LK only
updates velocity at the midpoint of its source frame interval. Detector
results are associated at their source frame and replayed through a bounded
history when they arrive late.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import time

import numpy as np


_DETECTOR_H = np.zeros((4, 6), dtype=float)
_DETECTOR_H[:4, :4] = np.eye(4)
_LK_H = np.zeros((2, 6), dtype=float)
_LK_H[0, 4] = _LK_H[1, 5] = 1.0
                                                                            
_DETECTOR_VELOCITY_D2 = 9.49
_DETECTOR_VELOCITY_CENTER_RATIO = .5
_DETECTOR_VELOCITY_SPEED_RATIO = 3.0                                 
_DUPLICATE_MIN_COVERAGE = .90
_DUPLICATE_MAX_AREA_RATIO = .60


def _finite_box(box):
    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    if values[2] - values[0] < 2 or values[3] - values[1] < 2:
        return None
    return values


def _measurement(box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return np.asarray(((x1 + x2) * .5, (y1 + y2) * .5,
                       math.log(width), math.log(height)), dtype=float)


def _state_box(state):
    cx, cy = float(state[0]), float(state[1])
    width = math.exp(max(-8.0, min(12.0, float(state[2]))))
    height = math.exp(max(-8.0, min(12.0, float(state[3]))))
    return cx - width * .5, cy - height * .5, cx + width * .5, cy + height * .5


def _clone_tracks(tracks):
    return {identity: track.clone() for identity, track in tracks.items()}


class DetectorNoiseModel:
    """Bucketed YOLO R_D loader with a visible, uncalibrated fallback."""

    CONFIDENCE_BINS = ((.15, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.000001))
    SHORT_SIDE_BINS = ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, float("inf")))

    def __init__(self, calibration_path=None):
        self.path = str(calibration_path) if calibration_path else None
        self.bins = []
        self.fallback_variance = None
        self.load_error = None
        if calibration_path:
            try:
                with open(calibration_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.bins = list(data.get("bins", []))
                fallback = data.get("fallback_variance")
                if fallback is not None:
                    self.fallback_variance = self._valid_variance(fallback)
            except (OSError, ValueError, TypeError) as exc:
                self.load_error = str(exc)

    @staticmethod
    def _valid_variance(values):
        array = np.asarray(values, dtype=float)
        if array.shape == (4,):
            array = np.diag(array)
        if array.shape != (4, 4) or not np.isfinite(array).all():
            return None
        array = (array + array.T) * .5
        if np.min(np.linalg.eigvalsh(array)) <= 0:
            return None
        return array

    def for_detection(self, detection):
        box = _finite_box(getattr(detection, "bbox", ()))
        if box is None:
            return np.diag((25.0, 25.0, .04, .04)), "fallback_invalid_box"
        short_side = min(box[2] - box[0], box[3] - box[1])
        confidence = float(getattr(detection, "confidence", 0.0))
        for item in self.bins:
            try:
                low_c, high_c = float(item["confidence_min"]), float(item["confidence_max"])
                low_s, high_s = float(item["short_side_min"]), float(item["short_side_max"])
                samples = int(item.get("samples", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if (samples >= 100 and low_c <= confidence < high_c
                    and low_s <= short_side < high_s):
                variance = self._valid_variance(item.get("covariance", item.get("variance")))
                if variance is not None:
                    return variance, "calibrated_%s" % item.get("name", "bucket")
        if self.fallback_variance is not None:
            return self.fallback_variance.copy(), "fallback_calibration_file"
                                                                             
        center_sigma = max(2.0, short_side * .08) / math.sqrt(max(.15, confidence))
        shape_sigma = .12 / math.sqrt(max(.15, confidence))
        return np.diag((center_sigma ** 2, center_sigma ** 2,
                        shape_sigma ** 2, shape_sigma ** 2)), "fallback_uncalibrated"

    def summary(self):
        calibrated_bins = sum(1 for item in self.bins
                              if int(item.get("samples", 0) or 0) >= 100
                              and self._valid_variance(item.get("covariance", item.get("variance"))) is not None)
        return dict(path=self.path, calibrated=calibrated_bins > 0,
                    calibrated_bucket_count=calibrated_bins,
                    fallback_present=self.fallback_variance is not None,
                    load_error=self.load_error,
                    confidence_bins=[list(pair) for pair in self.CONFIDENCE_BINS],
                    short_side_bins=[[lo, hi if math.isfinite(hi) else None]
                                     for lo, hi in self.SHORT_SIDE_BINS],
                    minimum_samples_per_bucket=100,
                    fallback_definition="center_sigma=max(2px,0.08*short_side)/sqrt(confidence); log-size_sigma=0.12/sqrt(confidence)")


class _TrackState:
    """Mutable per-track linear Kalman state; all times are monotonic seconds."""

    def __init__(self, identity, detection, timestamp, sequence, epoch, process_accel=180.0):
        box = _finite_box(getattr(detection, "bbox", ()))
        if box is None:
            raise ValueError("Cannot initialize a track from an invalid box")
        z = _measurement(box)
        short_side = min(box[2] - box[0], box[3] - box[1])
        self.track_id = identity
        self.birth_key = identity
        self.class_id = int(getattr(detection, "class_id", 0))
        self.x = np.asarray((z[0], z[1], z[2], z[3], 0.0, 0.0), dtype=float)
        center_var = max(4.0, (short_side * .12) ** 2)
        self.P = np.diag((center_var, center_var, .04, .04,
                          max(400.0, short_side ** 2 * 16.0),
                          max(400.0, short_side ** 2 * 16.0)))
        self.timestamp = float(timestamp)
        self.first_detection_at = float(timestamp)
        self.first_detection_sequence = int(sequence)
        self.last_detection_at = float(timestamp)
        self.last_semantic_confirmed_at = float(timestamp)
        self.last_detection_center = (float(z[0]), float(z[1]))
        self.last_lk_at = None
        self.last_detection_sequence = int(sequence)
        self.last_detection_confidence = float(getattr(detection, "confidence", 0.0))
        self.detection_records = [(int(sequence), self.last_detection_confidence, float(timestamp))]
        self.hits = 1
        self.misses = 0
        self.status = "tentative"
        self.last_innovation_d2 = None
        self.last_association_cost = None
        self.last_source = "detector"
        self.process_accel = float(process_accel)
        self.rd_source = "fallback_uncalibrated"
        self.lk_updates = 0
        self.last_lk_quality = {}

    def clone(self):
        other = object.__new__(_TrackState)
        other.__dict__ = dict(self.__dict__)
        other.x = self.x.copy()
        other.P = self.P.copy()
        other.detection_records = list(self.detection_records)
        other.last_lk_quality = dict(self.last_lk_quality)
        return other

    def predict_to(self, timestamp):
        timestamp = float(timestamp)
        dt = timestamp - self.timestamp
        if dt < -1e-9:
            raise ValueError("Kalman state cannot move backward in time")
        if dt <= 1e-12:
            self.timestamp = max(self.timestamp, timestamp)
            return
        F = np.eye(6, dtype=float)
        F[0, 4] = F[1, 5] = dt
        q = self.process_accel ** 2
        Q = np.zeros((6, 6), dtype=float)
        for position, velocity in ((0, 4), (1, 5)):
            Q[position, position] = q * dt ** 4 / 4.0
            Q[position, velocity] = Q[velocity, position] = q * dt ** 3 / 2.0
            Q[velocity, velocity] = q * dt ** 2
        Q[2, 2] = Q[3, 3] = .01 * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = (self.P + self.P.T) * .5
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all():
            raise FloatingPointError("Kalman prediction became non-finite")
        self.timestamp = timestamp

    def _update(self, z, H, R, *, preserve_velocity=False):
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)
        if not np.isfinite(z).all() or not np.isfinite(R).all():
            raise FloatingPointError("Kalman measurement is non-finite")
        residual = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError("Kalman innovation covariance is singular") from exc
        if preserve_velocity:
                                                                               
            K[4:, :] = 0.0
        self.x = self.x + K @ residual
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) * .5
        if not np.isfinite(self.x).all() or not np.isfinite(self.P).all():
            raise FloatingPointError("Kalman update became non-finite")

    def detector_update(self, detection, timestamp, sequence, R, rd_source, confirm_hits,
                        association_d2):
        box = _finite_box(getattr(detection, "bbox", ()))
        if box is None:
            raise ValueError("Invalid detector box")
        z = _measurement(box)
        cx, cy = float(z[0]), float(z[1])
        predicted_box = _state_box(self.x)
        short_side = max(2.0, min(box[2] - box[0], box[3] - box[1],
                                  predicted_box[2] - predicted_box[0], predicted_box[3] - predicted_box[1]))
        center_ratio = math.hypot(cx - self.x[0], cy - self.x[1]) / short_side
        dt = timestamp - self.last_detection_at
        speed_ratio = (math.hypot(cx - self.last_detection_center[0], cy - self.last_detection_center[1])
                       / (short_side * dt) if dt > 1e-9 else None)
        guard_reason = ""
        if association_d2 >= _DETECTOR_VELOCITY_D2:
            guard_reason = "high_innovation"
        elif (speed_ratio is not None and speed_ratio >= _DETECTOR_VELOCITY_SPEED_RATIO
              and center_ratio >= _DETECTOR_VELOCITY_CENTER_RATIO):
            guard_reason = "large_fast_center_jump"
        self._update(z, _DETECTOR_H, R, preserve_velocity=bool(guard_reason))
        self.last_innovation_d2 = float(association_d2)
        self.last_association_cost = self.last_innovation_d2
        self.last_detection_at = float(timestamp)
        self.last_semantic_confirmed_at = float(timestamp)
        self.last_detection_center = (cx, cy)
        self.last_detection_sequence = int(sequence)
        self.last_detection_confidence = float(getattr(detection, "confidence", 0.0))
        if not self.detection_records or self.detection_records[-1][0] != int(sequence):
            self.detection_records.append((int(sequence), self.last_detection_confidence, float(timestamp)))
        self.detection_records = self.detection_records[-4:]
        self.hits = len({record[0] for record in self.detection_records})
        self.misses = 0
        self.rd_source = rd_source
        self.last_source = "detector"
        if self.hits >= confirm_hits:
            self.status = "confirmed"
        return dict(detector_velocity_policy="position_size_only" if guard_reason else "kalman",
                    detector_velocity_guard_reason=guard_reason,
                    detector_center_innovation_ratio=center_ratio,
                    detector_anchor_speed_short_sides_s=speed_ratio)

    def lk_velocity_update(self, velocity, timestamp, R, quality):
        self.predict_to(timestamp)
        self._update(velocity, _LK_H, R)
        self.lk_updates += 1
        self.last_lk_at = float(timestamp)
        self.last_source = "lk_velocity"
        self.last_lk_quality = dict(quality or {})

    def refresh_lifecycle(self, timestamp, lost_after, retain_seconds,
                          tentative_timeout, confirm_hits=2):
        now = float(timestamp)
        detection_age = max(0.0, now - self.last_detection_at)
        if self.hits < confirm_hits:
            self.status = "deleted" if detection_age > tentative_timeout else "tentative"
            return
                                                                         
                                                                          
        if detection_age > retain_seconds:
            self.status = "deleted"
        elif detection_age > lost_after:
            self.status = "lost"
        else:
            self.status = "confirmed"


@dataclass(frozen=True)
class TrackView:
    track_id: str
    bbox: tuple
    confidence: float
    class_id: int
    status: str
    hits: int
    misses: int
    last_detection_at: float
    last_detection_sequence: int
    source: str
    position_source: str
    valid_until: float
    covariance: tuple
    velocity: tuple
    age_s: float
    last_visual_support_at: float
    visual_support_age_s: float
    identity_confirmed: bool = False
    last_innovation_d2: object = None
    last_association_cost: object = None
    rd_source: str = "fallback_uncalibrated"
    recent_detection_records: tuple = ()
    first_detection_sequence: object = None
    last_semantic_confirmed_at: object = None
    semantic_age_s: float = 0.0
    full_frame_unmatched_count: int = 0
    semantic_state: str = "verified"
    semantic_reason: str = ""
    semantic_valid_until: object = None


class _DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _hungarian(cost):
    """Rectangular Hungarian algorithm; rows <= columns; returns row/column pairs."""
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Assignment cost must be a matrix")
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return []
    if rows > columns:
        raise ValueError("Hungarian input requires rows <= columns")
    u = np.zeros(rows + 1, dtype=float)
    v = np.zeros(columns + 1, dtype=float)
    p = np.zeros(columns + 1, dtype=int)
    way = np.zeros(columns + 1, dtype=int)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(columns + 1, np.inf, dtype=float)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j], way[j] = current, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            if not math.isfinite(delta):
                break
            for j in range(columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, columns + 1):
        if p[j] != 0:
            pairs.append((int(p[j] - 1), int(j - 1)))
    return pairs


class MultiTargetTracker:
    """Single-writer multi-target KF tracker with bounded delayed-result replay."""

    def __init__(self, max_tracks=8, max_assignment_detections=16,
                 history_seconds=.5, max_history_events=128,
                 confirm_hits=2, birth_confidence=.15,
                 birth_center_separation_fraction=.5,
                 birth_center_max_sigma_fraction=.5,
                 lost_after_seconds=.25, retain_seconds=3.0,
                 tentative_timeout_seconds=1.0, prediction_horizon=.35,
                 mahalanobis_gate=18.47, size_ratio_min=.25, size_ratio_max=4.0,
                 process_accel=180.0, replay_budget_ms=12.0,
                 max_result_age=.35, rd_calibration_path=None,
                 tombstone_retention_seconds=1.0,
                 debug_association_matrix=False, epoch=0, detector_anchor_horizon=.75,
                 high_score_threshold=.35, low_score_threshold=.15,
                 hard_jump_center_ratio=2.0, hard_jump_speed_short_sides=6.0,
                 semantic_soft_age=1.2, semantic_hard_age=2.0,
                 semantic_miss_count=3):
        numeric = (history_seconds, lost_after_seconds, retain_seconds,
                   tentative_timeout_seconds, prediction_horizon, detector_anchor_horizon,
                   mahalanobis_gate, size_ratio_min, size_ratio_max,
                   process_accel, replay_budget_ms, birth_confidence,
                   birth_center_separation_fraction, birth_center_max_sigma_fraction,
                   max_result_age, tombstone_retention_seconds,
                   high_score_threshold, low_score_threshold,
                   hard_jump_center_ratio, hard_jump_speed_short_sides,
                   semantic_soft_age, semantic_hard_age)
        if (not all(math.isfinite(value) for value in numeric)
                or max_tracks < 1 or max_assignment_detections < 1 or max_history_events < 4
                or confirm_hits < 2 or history_seconds < .5 or lost_after_seconds <= 0
                or retain_seconds < lost_after_seconds or tentative_timeout_seconds <= 0
                or not 0 < prediction_horizon <= .35 or mahalanobis_gate <= 0
                or detector_anchor_horizon < prediction_horizon
                or not 0 < size_ratio_min <= 1 <= size_ratio_max or process_accel <= 0
                or replay_budget_ms <= 0 or not 0 <= birth_confidence <= 1
                or not 0 < birth_center_separation_fraction <= 1
                or not 0 < birth_center_max_sigma_fraction <= 1
                or max_result_age <= 0 or max_result_age > history_seconds
                or lost_after_seconds >= prediction_horizon
                or hard_jump_center_ratio <= 0 or hard_jump_speed_short_sides <= 0
                or semantic_soft_age <= 0 or semantic_hard_age <= semantic_soft_age
                or semantic_miss_count < 1):
            raise ValueError("Invalid multi-target tracker configuration")
        if not 0 <= low_score_threshold < high_score_threshold <= 1:
            raise ValueError("Require 0 <= low_score_threshold < high_score_threshold <= 1")
        if tombstone_retention_seconds < history_seconds:
            raise ValueError("Tombstone retention must be at least the history window")
        self.max_tracks = int(max_tracks)
        self.max_assignment_detections = int(max_assignment_detections)
        self.history_seconds = float(history_seconds)
        self.max_history_events = int(max_history_events)
        self.confirm_hits = int(confirm_hits)
        self.birth_confidence = float(birth_confidence)
        self.birth_center_separation_fraction = float(birth_center_separation_fraction)
        self.birth_center_max_sigma_fraction = float(birth_center_max_sigma_fraction)
        self.lost_after_seconds = float(lost_after_seconds)
        self.retain_seconds = float(retain_seconds)
        self.tentative_timeout_seconds = float(tentative_timeout_seconds)
        self.prediction_horizon = float(prediction_horizon)
        self.detector_anchor_horizon = float(detector_anchor_horizon)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.size_ratio_min, self.size_ratio_max = float(size_ratio_min), float(size_ratio_max)
        self.process_accel = float(process_accel)
        self.replay_budget_ms = float(replay_budget_ms)
        self.max_result_age = float(max_result_age)
        self.tombstone_retention_seconds = float(tombstone_retention_seconds)
        self.debug_association_matrix = bool(debug_association_matrix)
        self.high_score_threshold = float(high_score_threshold)
        self.low_score_threshold = float(low_score_threshold)
        self.hard_jump_center_ratio = float(hard_jump_center_ratio)
        self.hard_jump_speed_short_sides = float(hard_jump_speed_short_sides)
        self.semantic_soft_age = float(semantic_soft_age)
        self.semantic_hard_age = float(semantic_hard_age)
        self.semantic_miss_count = int(semantic_miss_count)
        self.noise_model = DetectorNoiseModel(rd_calibration_path)
        self.stream_epoch = int(epoch)
        self.active_track_id = None
        self.reset(epoch=epoch)

    def reset(self, epoch=None):
        if epoch is not None:
            self.stream_epoch = int(epoch)
        self.tracks = {}
        self._retired_tracks = {}
        self._events = []
        self._semantic_events = {}
        self._boundaries = OrderedDict()
        self._latest_timestamp = None
        self._latest_sequence = None
        self._frame_shape = None
        self._last_diagnostic = {}
        self._capacity_was_full = False
        self._capacity_last_log_timestamp = None
        self._capacity_rejections_pending = 0
        self._capacity_rejection_keys = set()
        self._counters = dict(accepted_detections=0, rejected_detections=0,
                              stale_results=0, duplicate_results=0, tracks_created=0,
                              tracks_deleted=0, replay_count=0, replay_rejected=0,
                              replay_fallback_full=0, candidate_columns_limited=0,
                              lk_velocity_updates=0, lk_velocity_rejected=0,
                              high_score_candidates=0, low_score_candidates=0,
                              high_score_matches=0, low_score_matches=0,
                              low_score_unmatched=0)

    def set_active_track_id(self, identity):
        self.active_track_id = identity

    def contains_track(self, identity):
        """Return whether an ID is still in the lifecycle pool, even if its box is invalid."""
        return identity is not None and str(identity) in self.tracks

    def capacity_log_state(self, timestamp, rejected_candidates=0):
        """Rate-limit capacity diagnostics to entry and 5-second summaries."""
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Capacity log timestamp must be finite")
        self._capacity_rejections_pending += max(0, int(rejected_candidates))
        full = len(self.tracks) >= self.max_tracks
        event = ""
        report_count = 0
        if not full:
            if self._capacity_was_full:
                event = "became_available"
                report_count = self._capacity_rejections_pending
            self._capacity_was_full = False
            self._capacity_last_log_timestamp = None
            self._capacity_rejections_pending = 0
        elif not self._capacity_was_full:
            event = "entered_full"
            report_count = self._capacity_rejections_pending
            self._capacity_rejections_pending = 0
            self._capacity_last_log_timestamp = timestamp
            self._capacity_was_full = True
        elif self._capacity_last_log_timestamp is None:
            event = "entered_full"
            report_count = self._capacity_rejections_pending
            self._capacity_rejections_pending = 0
            self._capacity_last_log_timestamp = timestamp
        elif timestamp - self._capacity_last_log_timestamp >= 5.0:
            event = "still_full"
            report_count = self._capacity_rejections_pending
            self._capacity_rejections_pending = 0
            self._capacity_last_log_timestamp = timestamp
        return dict(full=full, event=event, rejected_candidates=report_count,
                    track_count=len(self.tracks), max_tracks=self.max_tracks)

    @staticmethod
    def _clone_snapshot(snapshot):
        return _clone_tracks(snapshot)

    def _predicted_clone(self, track, timestamp):
        result = track.clone()
        result.predict_to(timestamp)
        return result

    def _association_cost(self, track, detection, timestamp):
        box = _finite_box(getattr(detection, "bbox", ()))
        if box is None or int(getattr(detection, "class_id", -1)) != track.class_id:
            return None, "class_or_box_gate"
        predicted = self._predicted_clone(track, timestamp)
        predicted_box = _state_box(predicted.x)
        det_width, det_height = box[2] - box[0], box[3] - box[1]
        pred_width, pred_height = predicted_box[2] - predicted_box[0], predicted_box[3] - predicted_box[1]
        dt = float(timestamp) - track.last_detection_at
        if track.hits >= self.confirm_hits and dt > 1e-6:
            measurement = _measurement(box)
                                                                          
                                                                            
                                                                            
            displacement = math.hypot(float(measurement[0] - track.last_detection_center[0]),
                                      float(measurement[1] - track.last_detection_center[1]))
            short_side = max(2.0, min(det_width, det_height, pred_width, pred_height))
            center_ratio = displacement / short_side
            speed_short_sides = displacement / (short_side * dt)
            if (center_ratio >= self.hard_jump_center_ratio
                    and speed_short_sides >= self.hard_jump_speed_short_sides):
                return None, "large_fast_center_jump"
        width_ratio, height_ratio = det_width / max(pred_width, 1e-6), det_height / max(pred_height, 1e-6)
        if (not self.size_ratio_min <= width_ratio <= self.size_ratio_max
                or not self.size_ratio_min <= height_ratio <= self.size_ratio_max):
            return None, "size_ratio_gate"
        R, _ = self.noise_model.for_detection(detection)
        S = _DETECTOR_H @ predicted.P @ _DETECTOR_H.T + R
        residual = _measurement(box) - _DETECTOR_H @ predicted.x
        try:
            d2 = float(residual.T @ np.linalg.solve(S, residual))
        except np.linalg.LinAlgError:
            return None, "singular_innovation"
        if not math.isfinite(d2) or d2 > self.mahalanobis_gate:
            return None, "mahalanobis_gate"
        return d2, "gated"

    def _valid_track_ids(self, tracks, timestamp):
        result = []
        for identity, track in tracks.items():
            track.refresh_lifecycle(timestamp, self.lost_after_seconds,
                                    self.retain_seconds, self.tentative_timeout_seconds,
                                    self.confirm_hits)
            if track.status != "deleted":
                result.append(identity)
        return result

    def _near_confirmed_for_birth(self, track, bbox, timestamp, class_id=None):
        """Return proximity plus axis sigmas for a same-class confirmed track.

        The conservative uncertainty radius is max(sigma_x, sigma_y). If that
        radius exceeds the configured fraction of the predicted short side,
        the duplicate guard is skipped and the candidate is allowed to spawn.
        """
        if (track.status != "confirmed"
                or (class_id is not None and track.class_id != int(class_id))):
            return False, None, None, None, None, False
        predicted = self._predicted_clone(track, timestamp)
        track_box = _state_box(predicted.x)
        width, height = track_box[2] - track_box[0], track_box[3] - track_box[1]
        short_side = min(width, height)
        if not math.isfinite(short_side) or short_side <= 0:
            return False, None, None, None, None, True
        track_cx = (track_box[0] + track_box[2]) * .5
        track_cy = (track_box[1] + track_box[3]) * .5
        det_cx = (bbox[0] + bbox[2]) * .5
        det_cy = (bbox[1] + bbox[3]) * .5
        distance = math.hypot(det_cx - track_cx, det_cy - track_cy)
        variance = np.diag(predicted.P)[:2]
        if (variance.shape != (2,) or not np.isfinite(variance).all()
                or np.min(variance) < 0):
            return False, distance, None, None, None, True
        sigma_x, sigma_y = math.sqrt(float(variance[0])), math.sqrt(float(variance[1]))
        position_sigma = max(sigma_x, sigma_y)
        sigma_limit = short_side * self.birth_center_max_sigma_fraction
        if position_sigma > sigma_limit:
            return False, distance, position_sigma, sigma_x, sigma_y, True
        limit = short_side * self.birth_center_separation_fraction
        return distance <= limit, distance, position_sigma, sigma_x, sigma_y, False

    def _normalize_detections(self, detections):
        normalized = []
        invalid = 0
        for index, detection in enumerate(detections or ()):
            box = _finite_box(getattr(detection, "bbox", ()))
            try:
                confidence = float(getattr(detection, "confidence", 0.0))
                class_id = int(getattr(detection, "class_id", 0))
            except (TypeError, ValueError):
                invalid += 1
                continue
            if box is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                invalid += 1
                continue
            normalized.append(dict(bbox=box, confidence=confidence,
                                   class_id=class_id, source_index=index,
                                   original=detection))
        return normalized, invalid

    def _limit_detections(self, detections, tracks, timestamp):
        if len(detections) <= self.max_assignment_detections:
            return detections, 0, []
        ranked = []
        for item in detections:
            best = math.inf
            for track in tracks.values():
                cost, _ = self._association_cost(track, item["original"], timestamp)
                if cost is not None:
                    best = min(best, cost)
            ranked.append((0 if math.isfinite(best) else 1, best,
                           -item["confidence"], item["source_index"], item))
        ranked.sort(key=lambda row: row[:4])
        limited = [row[-1] for row in ranked[:self.max_assignment_detections]]
        limited.sort(key=lambda item: item["source_index"])
        limited_indices = {item["source_index"] for item in limited}
        rejected = [item for item in detections if item["source_index"] not in limited_indices]
        return limited, len(rejected), rejected

    def _contained_duplicates(self, detections):
        """Find smaller same-class boxes covered by a retained, birth-eligible box."""
        if len(detections) < 2:
            return {}
        def area(item):
            x1, y1, x2, y2 = item["bbox"]
            return (x2 - x1) * (y2 - y1)

        ranked = sorted(detections, key=lambda item: (
            -area(item), -item["confidence"], item["bbox"], item["source_index"]))
        parents = []
        duplicates = {}
        for item in ranked:
            x1, y1, x2, y2 = item["bbox"]
            item_area = area(item)
            for parent, parent_area in parents:
                if (parent["class_id"] != item["class_id"]
                        or item_area > parent_area * _DUPLICATE_MAX_AREA_RATIO):
                    continue
                px1, py1, px2, py2 = parent["bbox"]
                overlap = (max(0.0, min(x2, px2) - max(x1, px1))
                           * max(0.0, min(y2, py2) - max(y1, py1)))
                if overlap >= item_area * _DUPLICATE_MIN_COVERAGE:
                    duplicates[item["source_index"]] = parent["source_index"]
                    break
            else:
                if item["confidence"] >= self.birth_confidence:
                    parents.append((item, item_area))
        return duplicates

    def _make_room(self, tracks, timestamp, outside_count=0):
        """Prune only lifecycle-expired tracks; never evict a still-retained ID."""
        pruned = []
        for identity, track in list(tracks.items()):
            track.refresh_lifecycle(timestamp, self.lost_after_seconds,
                                    self.retain_seconds, self.tentative_timeout_seconds,
                                    self.confirm_hits)
            if track.status == "deleted":
                tracks.pop(identity, None)
                pruned.append(identity)
        return len(tracks) + int(outside_count) < self.max_tracks, pruned

    def _can_make_room(self, tracks, timestamp, outside_count=0):
        """Check capacity before birth-quality gates without mutating track state."""
        total = len(tracks) + int(outside_count)
        if total < self.max_tracks:
            return True
        expired = 0
        for track in tracks.values():
            candidate = track.clone()
            candidate.refresh_lifecycle(
                timestamp, self.lost_after_seconds, self.retain_seconds,
                self.tentative_timeout_seconds, self.confirm_hits)
            if candidate.status == "deleted":
                expired += 1
        return total - expired < self.max_tracks

    def _apply_detector(self, tracks, event, outside_count=0):
        timestamp, sequence = event["timestamp"], event["sequence"]
        event["assignments"] = {}
        event["rejections"] = {}
        event["retired_ids"] = []
        all_detections, invalid = self._normalize_detections(event["detections"])
        live_ids = self._valid_track_ids(tracks, timestamp)
        pruned_ids = []
        for identity in tuple(tracks):
            if tracks[identity].status == "deleted":
                tracks.pop(identity, None)
                pruned_ids.append(identity)
        live_tracks = {identity: tracks[identity] for identity in live_ids}
        detections, limited, limited_items = self._limit_detections(
            all_detections, live_tracks, timestamp)
        for item in limited_items:
            event["rejections"][item["source_index"]] = "candidate_limit"
        duplicates = self._contained_duplicates(detections)
        for source_index in duplicates:
            event["rejections"][source_index] = "contained_duplicate"
        if duplicates:
            detections = [item for item in detections if item["source_index"] not in duplicates]

        high_detections, low_detections = [], []
        score_stage = {}
        for item in detections:
            index = item["source_index"]
            if item["confidence"] >= self.high_score_threshold:
                item["score_stage"] = "high"
                high_detections.append(item)
                score_stage[index] = "high"
            elif item["confidence"] >= self.low_score_threshold:
                item["score_stage"] = "low"
                low_detections.append(item)
                score_stage[index] = "low"
            else:
                item["score_stage"] = "below_low"
                score_stage[index] = "below_low"
                event["rejections"].setdefault(index, "below_low_score_threshold")

        track_ids = list(live_tracks)
        assignment_costs = {}
        detector_update_details = {}
        pair_candidates = {}
        birth_guard_matches = {}
        birth_guard_evaluation = {}
        pairwise_costs = {}
        pairwise_rejections = {}
        matched_track_ids = set()
        matched_detection_indices = set()
        assigned_stage = {}

        def associate(stage_detections, stage_track_ids, stage_name):
            stage_tracks = [identity for identity in stage_track_ids
                            if identity not in matched_track_ids]
            stage_items = [item for item in stage_detections
                           if item["source_index"] not in matched_detection_indices]
            if not stage_tracks or not stage_items:
                return
            unmatched_cost = self.mahalanobis_gate
            matrix = np.full((len(stage_tracks), len(stage_items) + len(stage_tracks)),
                             unmatched_cost, dtype=float)
            matrix[:, :len(stage_items)] = unmatched_cost + 1e6
            for row, identity in enumerate(stage_tracks):
                for col, item in enumerate(stage_items):
                    cost, reason = self._association_cost(
                        live_tracks[identity], item["original"], timestamp)
                    source_key = str(item["source_index"])
                    pair_candidates.setdefault(source_key, []).append((cost, str(identity), reason))
                    if self.debug_association_matrix:
                        pairwise_costs.setdefault(source_key, {})[str(identity)] = cost
                    if cost is not None:
                        matrix[row, col] = cost
                    elif self.debug_association_matrix:
                        pairwise_rejections.setdefault(source_key, {})[str(identity)] = reason
            for row, col in _hungarian(matrix):
                if col >= len(stage_items) or matrix[row, col] >= unmatched_cost:
                    continue
                item = stage_items[col]
                identity = stage_tracks[row]
                R, rd_source = self.noise_model.for_detection(item["original"])
                live_tracks[identity].predict_to(timestamp)
                detector_update_details[item["source_index"]] = live_tracks[identity].detector_update(
                    item["original"], timestamp, sequence, R, rd_source, self.confirm_hits,
                    association_d2=matrix[row, col])
                tracks[identity] = live_tracks[identity]
                matched_track_ids.add(identity)
                matched_detection_indices.add(item["source_index"])
                assigned_stage[item["source_index"]] = stage_name
                assignment_costs[item["source_index"]] = float(matrix[row, col])
                event["assignments"][item["source_index"]] = identity

                                                                      
        associate(high_detections, track_ids, "high_match")
                                                                               
                                                                            
        associate(low_detections, track_ids, "low_match")

        for identity in track_ids:
            if identity not in matched_track_ids:
                live_tracks[identity].misses += 1
                live_tracks[identity].refresh_lifecycle(timestamp, self.lost_after_seconds,
                                                        self.retain_seconds, self.tentative_timeout_seconds,
                                                        self.confirm_hits)
                tracks[identity] = live_tracks[identity]

        created = evicted = 0
        birth_detections = [item for item in high_detections + low_detections
                            if item["source_index"] not in matched_detection_indices]
        for item in birth_detections:
            if item["confidence"] < self.birth_confidence:
                event["rejections"][item["source_index"]] = "birth_confidence_low"
                continue
                                                                               
            if not self._can_make_room(tracks, timestamp, outside_count):
                event["rejections"][item["source_index"]] = "track_capacity"
                birth_guard_evaluation[item["source_index"]] = dict(
                    evaluated=False, skip_reason="capacity_preflight")
                continue
            birth_guard_evaluation[item["source_index"]] = dict(evaluated=True, skip_reason="")
            near_track = None
            near_distance = near_sigma = near_sigma_x = near_sigma_y = None
            nearest_uncertain = None
            for identity, track in live_tracks.items():
                if track.class_id != item["class_id"]:
                    continue
                (is_near, distance, position_sigma, sigma_x, sigma_y,
                 skipped_uncertain) = self._near_confirmed_for_birth(
                    track, item["bbox"], timestamp, item["class_id"])
                if is_near:
                    near_track, near_distance, near_sigma = str(identity), distance, position_sigma
                    near_sigma_x, near_sigma_y = sigma_x, sigma_y
                    break
                if skipped_uncertain and distance is not None and (
                        nearest_uncertain is None or distance < nearest_uncertain[1]):
                    nearest_uncertain = (str(identity), distance, position_sigma, sigma_x, sigma_y)
            if near_track is not None:
                event["rejections"][item["source_index"]] = "birth_near_confirmed"
                birth_guard_matches[item["source_index"]] = dict(
                    track_id=near_track, center_distance_px=float(near_distance),
                    position_sigma_px=float(near_sigma),
                    position_sigma_x_px=float(near_sigma_x),
                    position_sigma_y_px=float(near_sigma_y), skipped_uncertain=False)
                continue
            if nearest_uncertain is not None:
                birth_guard_evaluation[item["source_index"]]["skip_reason"] = "position_uncertainty"
                birth_guard_matches[item["source_index"]] = dict(
                    track_id=nearest_uncertain[0], center_distance_px=float(nearest_uncertain[1]),
                    position_sigma_px=(float(nearest_uncertain[2]) if nearest_uncertain[2] is not None else None),
                    position_sigma_x_px=(float(nearest_uncertain[3]) if nearest_uncertain[3] is not None else None),
                    position_sigma_y_px=(float(nearest_uncertain[4]) if nearest_uncertain[4] is not None else None),
                    skipped_uncertain=True)
            if len(tracks) + outside_count >= self.max_tracks:
                room, expired = self._make_room(tracks, timestamp, outside_count=outside_count)
                evicted += len(expired)
                pruned_ids.extend(expired)
            else:
                room = True
            if not room:
                event["rejections"][item["source_index"]] = "track_capacity"
                continue
            identity = "%s:%s:%s" % (self.stream_epoch, sequence, item["source_index"])
            if identity in tracks:
                event["rejections"][item["source_index"]] = "stable_birth_key_collision"
                continue
            tracks[identity] = _TrackState(identity, item["original"], timestamp, sequence,
                                           self.stream_epoch, self.process_accel)
            event["assignments"][item["source_index"]] = identity
            assigned_stage[item["source_index"]] = "%s_birth" % score_stage[item["source_index"]]
            created += 1

        for item in low_detections:
            if (item["source_index"] not in matched_detection_indices
                    and item["source_index"] not in event["assignments"]):
                event["rejections"].setdefault(
                    item["source_index"], "low_score_unmatched_no_birth")

        detection_summaries = {}
        for item in all_detections:
            source_index = item["source_index"]
            source_key = str(source_index)
            ranked = sorted((pair for pair in pair_candidates.get(source_key, ())
                             if pair[0] is not None), key=lambda pair: (pair[0], pair[1]))
            best = ranked[0] if ranked else None
            second = ranked[1] if len(ranked) > 1 else None
            assigned = event["assignments"].get(source_index)
            associated_existing = assigned in track_ids if assigned is not None else False
            birth_guard = birth_guard_evaluation.get(source_index, {})
            gate_counts = {}
            for _, _, gate_reason in pair_candidates.get(source_key, ()):
                if gate_reason != "gated":
                    gate_counts[gate_reason] = gate_counts.get(gate_reason, 0) + 1
            if assigned is not None:
                rejection_reason = ("matched_existing" if associated_existing else "new_track")
            else:
                rejection_reason = event["rejections"].get(source_index)
                if rejection_reason is None:
                    rejection_reason = "hungarian_unmatched" if best is not None else "no_valid_track_gate"
            detection_summaries[source_key] = dict(
                source_index=source_index,
                confidence=float(item["confidence"]),
                score_stage=score_stage.get(source_index, "not_associated"),
                association_stage=assigned_stage.get(source_index, "unmatched"),
                best_track_id=best[1] if best else None,
                best_cost=float(best[0]) if best else None,
                second_track_id=second[1] if second else None,
                second_cost=float(second[0]) if second else None,
                best_second_margin=float(second[0] - best[0]) if best and second else None,
                assigned_track_id=str(assigned) if assigned is not None else None,
                association_success=bool(associated_existing),
                track_created=bool(assigned is not None and not associated_existing),
                rejection_reason=rejection_reason,
                duplicate_parent_source_index=duplicates.get(source_index),
                birth_guard_track_id=birth_guard_matches.get(source_index, {}).get("track_id"),
                birth_guard_center_distance_px=birth_guard_matches.get(source_index, {}).get("center_distance_px"),
                birth_guard_position_sigma_px=birth_guard_matches.get(source_index, {}).get("position_sigma_px"),
                birth_guard_position_sigma_x_px=birth_guard_matches.get(source_index, {}).get("position_sigma_x_px"),
                birth_guard_position_sigma_y_px=birth_guard_matches.get(source_index, {}).get("position_sigma_y_px"),
                birth_guard_evaluated=bool(birth_guard.get("evaluated", False)),
                birth_guard_skip_reason=birth_guard.get("skip_reason", ""),
                birth_guard_skipped_uncertain=(
                    bool(birth_guard_matches.get(source_index, {}).get("skipped_uncertain", False))
                    if birth_guard.get("evaluated", False) else None),
                gate_rejection_counts=gate_counts,
                **detector_update_details.get(source_index, {}))
        event["retired_ids"] = sorted(set(pruned_ids))
        event["diagnostics"] = dict(
            raw_count=len(event["detections"] or ()), valid_count=len(all_detections),
            invalid_count=invalid, limited_count=limited,
            high_score_count=len(high_detections), low_score_count=len(low_detections),
            high_score_matches=sum(1 for value in assigned_stage.values() if value == "high_match"),
            low_score_matches=sum(1 for value in assigned_stage.values() if value == "low_match"),
            low_score_unmatched=sum(1 for item in low_detections
                                    if item["source_index"] not in matched_detection_indices),
            candidate_costs=assignment_costs, detection_summaries=detection_summaries,
            pairwise_costs=pairwise_costs if self.debug_association_matrix else {},
            pairwise_rejections=pairwise_rejections if self.debug_association_matrix else {},
            rejections=dict(event["rejections"]), tracks_created=created,
            tracks_evicted=evicted, capacity_pruned_ids=list(event["retired_ids"]))
        event["applied"] = True
        return event

    @staticmethod
    def _copy_event(event):
        copied = dict(event)
        for key in ("assignments", "rejections", "diagnostics"):
            copied[key] = dict(event.get(key, {}))
        copied["retired_ids"] = list(event.get("retired_ids", ()))
        return copied

    def _advance_lifecycle(self, tracks, timestamp):
        deleted = []
        for identity, track in tracks.items():
            track.refresh_lifecycle(timestamp, self.lost_after_seconds,
                                    self.retain_seconds, self.tentative_timeout_seconds,
                                    self.confirm_hits)
            if track.status == "deleted":
                deleted.append(identity)
        for identity in deleted:
            tracks.pop(identity, None)
        return deleted

    def _filter_retired(self, tracks, timestamp, filtered_ids=None):
        for identity, retired_at in self._retired_tracks.items():
            if retired_at <= timestamp + 1e-9 and identity in tracks:
                tracks.pop(identity, None)
                if filtered_ids is not None:
                    filtered_ids.add(str(identity))
        return tracks

    def advance(self, timestamp, sequence, lk_observations=(), frame_shape=None):
        timestamp = float(timestamp)
        sequence = int(sequence)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Frame timestamp must be finite and nonnegative")
        if self._latest_timestamp is not None and timestamp <= self._latest_timestamp:
            raise ValueError("Frame timestamps must be strictly increasing")
        if self._latest_sequence is not None and sequence <= self._latest_sequence:
            raise ValueError("Frame sequence must be strictly increasing")
        if self._frame_shape is not None and frame_shape is not None and tuple(frame_shape) != self._frame_shape:
            self.reset(epoch=self.stream_epoch + 1)
        prior_time = self._latest_timestamp
        current_tracks = _clone_tracks(self.tracks)
        accepted_lk = []
        for observation in lk_observations or ():
            try:
                t0, t1 = float(observation["t0"]), float(observation["t1"])
                measurement_at = float(observation["measurement_at"])
                identity = observation["track_id"]
                z = tuple(float(value) for value in observation["z"])
                R = np.asarray(observation["R"], dtype=float)
                if (prior_time is None or abs(t0 - prior_time) > 1e-6
                        or abs(t1 - timestamp) > 1e-6 or abs(measurement_at - (t0 + t1) * .5) > 1e-6
                        or len(z) != 2 or R.shape != (2, 2) or not np.isfinite(R).all()
                        or np.min(np.linalg.eigvalsh((R + R.T) * .5)) <= 0):
                    raise ValueError("invalid_interval_or_covariance")
                track = current_tracks.get(identity)
                if track is None or track.status == "deleted":
                    raise ValueError("unknown_track")
                track.predict_to(measurement_at)
                track.lk_velocity_update(z, measurement_at, R, observation.get("quality", {}))
                track.predict_to(timestamp)
                accepted_lk.append(dict(kind="lk", timestamp=measurement_at,
                                        sequence=sequence, track_id=identity,
                                        sequence0=int(observation["sequence0"]),
                                        sequence1=int(observation["sequence1"]),
                                        t0=t0, t1=t1, measurement_at=measurement_at,
                                        z=z, R=R.copy(), quality=dict(observation.get("quality", {})),
                                        active_epoch=int(observation.get("active_epoch", 0))))
                self._counters["lk_velocity_updates"] += 1
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                self._counters["lk_velocity_rejected"] += 1
                self._last_diagnostic["lk_reject_reason"] = str(exc)
        for track in current_tracks.values():
            track.predict_to(timestamp)
        retired_ids = self._advance_lifecycle(current_tracks, timestamp)
        self.tracks = current_tracks
        for identity in retired_ids:
            self._retired_tracks.setdefault(identity, timestamp)
        self._latest_timestamp, self._latest_sequence = timestamp, sequence
        if frame_shape is not None:
            self._frame_shape = tuple(frame_shape)
        for event in accepted_lk:
            self._events.append(event)
        self._events.append(dict(kind="frame", timestamp=timestamp, sequence=sequence))
        snapshot = _clone_tracks(self.tracks)
        self._boundaries[sequence] = dict(timestamp=timestamp, sequence=sequence,
                                           pre=snapshot, post=_clone_tracks(snapshot))
        self._trim_history()
        self._last_diagnostic = dict(reason="advanced", sequence=sequence,
                                     timestamp=timestamp, lk_accepted=len(accepted_lk),
                                     track_count=len(self.tracks), replay_depth=0,
                                     replay_scope="none", replay_ms=0.0)
        return self.snapshot(timestamp)

    def _event_order(self, event):
        priority = {"frame": 0, "detector": 1, "lk": 2}
        return event["timestamp"], priority.get(event["kind"], 9), event.get("sequence", -1)

    def _detector_events(self, events=None):
        return [event for event in (self._events if events is None else events)
                if event["kind"] == "detector"]

    def _semantic_evidence(self, track, timestamp):
        confirmed_at = track.last_semantic_confirmed_at
        age = max(0.0, float(timestamp) - confirmed_at)
        misses = sum(1 for event in self._semantic_events.values()
                     if confirmed_at + 1e-9 < event["timestamp"] <= timestamp + 1e-9
                     and track.track_id not in event["assigned_track_ids"])
        if age >= self.semantic_hard_age:
            return age, misses, "invalid", "hard_age"
        if age >= self.semantic_soft_age and misses >= self.semantic_miss_count:
            return age, misses, "suspect", "full_frame_misses_and_age"
        return age, misses, "verified", "fresh_detector_or_short_gap"

    def _candidate_track_ids(self, event, snapshot):
        detections, _ = self._normalize_detections(event["detections"])
        result = set()
        for item in detections:
            for identity, track in snapshot.items():
                if track.status == "deleted":
                    continue
                cost, _ = self._association_cost(track, item["original"], event["timestamp"])
                if cost is not None:
                    result.add(identity)
        return result

    def _discover_component(self, new_event, source_boundary):
        """One bounded source-snapshot graph scan; union-find closure is transitive."""
        dsu = _DisjointSet()
        detector_events = self._detector_events() + [new_event]
        packet_nodes = {}
        for event in detector_events:
                                                                          
                                                                             
                                                                              
            packet = ("packet", int(event["sequence"]))
            packet_nodes[int(event["sequence"])] = packet
            dsu.find(packet)
            detections, _ = self._normalize_detections(event["detections"])
            for item in detections:
                det_node = ("detection", int(event["sequence"]), int(item["source_index"]))
                dsu.union(packet, det_node)
                boundary = self._boundaries.get(int(event["sequence"]))
                pre = boundary["pre"] if boundary is not None else {}
                for identity, track in pre.items():
                    if track.status == "deleted":
                        continue
                    cost, rejection_reason = self._association_cost(
                        track, item["original"], event["timestamp"])
                    birth_near = (item["confidence"] >= self.birth_confidence
                                  and self._near_confirmed_for_birth(
                                      track, item["bbox"], event["timestamp"],
                                      item["class_id"])[0])
                                                                           
                                                                             
                                                                             
                    if (cost is not None or birth_near
                            or rejection_reason == "large_fast_center_jump"):
                        dsu.union(det_node, ("track", identity))
                assigned = event.get("assignments", {}).get(item["source_index"])
                if assigned is not None:
                    dsu.union(det_node, ("track", assigned))
                if event is new_event and item["confidence"] >= self.birth_confidence:
                    birth = "%s:%s:%s" % (self.stream_epoch, event["sequence"], item["source_index"])
                    dsu.union(det_node, ("track", birth))
            for identity in event.get("retired_ids", ()):
                dsu.union(packet, ("track", identity))
        seed_root = dsu.find(packet_nodes[int(new_event["sequence"])])
        affected_sequences = {seq for seq, node in packet_nodes.items() if dsu.find(node) == seed_root}
        affected_tracks = {node[1] for node in dsu.parent
                           if isinstance(node, tuple) and len(node) == 2
                           and node[0] == "track" and dsu.find(node) == seed_root}
                                                                                            
        return affected_tracks, affected_sequences

    def _merge_snapshot(self, old_snapshot, replacement, identities):
        merged = _clone_tracks(old_snapshot)
        for identity in identities:
            merged.pop(identity, None)
            if identity in replacement:
                merged[identity] = replacement[identity].clone()
        return merged

    def _replay(self, source_sequence, new_event, affected_tracks,
                affected_sequences, full=False):
        source_boundary = self._boundaries[source_sequence]
        seed = source_boundary["pre"]
        if full:
            identities = set(seed)
            replay_tracks = _clone_tracks(seed)
        else:
            identities = set(affected_tracks)
            replay_tracks = {identity: seed[identity].clone()
                             for identity in identities if identity in seed}
        tombstone_filtered_ids = set()
                                                                             
                                                                             
        self._filter_retired(replay_tracks, source_boundary["timestamp"],
                             filtered_ids=tombstone_filtered_ids)
        replay_events = [event for event in self._events
                         if (event["timestamp"] > source_boundary["timestamp"] + 1e-9
                             or (event["kind"] == "detector"
                                 and event["sequence"] in affected_sequences
                                 and event["timestamp"] >= source_boundary["timestamp"] - 1e-9))]
        replay_events = [event for event in replay_events if event["kind"] != "detector"
                         or event["sequence"] != source_sequence]
        replay_events.append(new_event)
        replay_events.sort(key=self._event_order)
        resulting_pre, resulting_post, assignment_updates = {}, {}, {}
        retirements = []
        start_time = source_boundary["timestamp"]
        start_clock = time.perf_counter()
        cross_component_edge = False
                                                                                        
        inserted = [event for event in replay_events if event["kind"] == "detector"
                    and event["sequence"] == source_sequence]
        for event in inserted:
            self._filter_retired(replay_tracks, event["timestamp"],
                                 filtered_ids=tombstone_filtered_ids)
            if not full and self._has_cross_component_edge(event, replay_tracks, identities):
                cross_component_edge = True
                break
            outside_count = self._outside_count(source_sequence, identities, full)
            applied = self._apply_detector(replay_tracks, self._copy_event(event),
                                           outside_count=outside_count)
            assignment_updates[event["sequence"]] = applied
            retirements.extend((identity, event["timestamp"])
                               for identity in applied.get("retired_ids", ()))
            resulting_post[source_sequence] = _clone_tracks(replay_tracks)
            if (not full and outside_count
                    and "track_capacity" in applied.get("rejections", {}).values()):
                cross_component_edge = True
                break
        if cross_component_edge:
            return None, "cross_component_edge", 0, 0.0
        depth = len(replay_events)
        for event in replay_events:
            if event["kind"] == "detector" and event["sequence"] == source_sequence:
                continue
            if event["timestamp"] <= start_time + 1e-9:
                continue
                                                                               
                                                                              
                                                                     
            self._filter_retired(replay_tracks, event["timestamp"],
                                 filtered_ids=tombstone_filtered_ids)
            if event["kind"] == "frame":
                for track in replay_tracks.values():
                    track.predict_to(event["timestamp"])
                deleted = self._advance_lifecycle(replay_tracks, event["timestamp"])
                retirements.extend((identity, event["timestamp"]) for identity in deleted)
                resulting_pre[event["sequence"]] = _clone_tracks(replay_tracks)
            elif event["kind"] == "lk":
                track = replay_tracks.get(event["track_id"])
                if track is not None:
                    track.predict_to(event["measurement_at"])
                    track.lk_velocity_update(event["z"], event["measurement_at"],
                                             event["R"], event["quality"])
            elif event["kind"] == "detector" and event["sequence"] in affected_sequences:
                if not full and self._has_cross_component_edge(event, replay_tracks, identities):
                    cross_component_edge = True
                    break
                outside_count = self._outside_count(event["sequence"], identities, full)
                applied = self._apply_detector(replay_tracks, self._copy_event(event),
                                               outside_count=outside_count)
                assignment_updates[event["sequence"]] = applied
                retirements.extend((identity, event["timestamp"])
                                   for identity in applied.get("retired_ids", ()))
                resulting_post[event["sequence"]] = _clone_tracks(replay_tracks)
                if (not full and outside_count
                        and "track_capacity" in applied.get("rejections", {}).values()):
                    cross_component_edge = True
                    break
            elif event["kind"] == "detector" and not full:
                if self._has_component_edge(event, replay_tracks):
                    cross_component_edge = True
                    break
            if (time.perf_counter() - start_clock) * 1000.0 > self.replay_budget_ms:
                return None, "replay_budget_exceeded", depth, (time.perf_counter() - start_clock) * 1000.0
        if cross_component_edge:
            return None, "cross_component_edge", depth, (time.perf_counter() - start_clock) * 1000.0
        for track in replay_tracks.values():
            track.predict_to(self._latest_timestamp)
        deleted = self._advance_lifecycle(replay_tracks, self._latest_timestamp)
        retirements.extend((identity, self._latest_timestamp) for identity in deleted)
        self._filter_retired(replay_tracks, self._latest_timestamp,
                             filtered_ids=tombstone_filtered_ids)
        elapsed_ms = (time.perf_counter() - start_clock) * 1000.0
        if elapsed_ms > self.replay_budget_ms:
            return None, "replay_budget_exceeded", depth, elapsed_ms
        return dict(tracks=replay_tracks, pre=resulting_pre, post=resulting_post,
                    assignments=assignment_updates, identities=identities,
                    retirements=retirements,
                    tombstone_filtered_track_ids=sorted(tombstone_filtered_ids)),\
            "accepted", depth, elapsed_ms

    def _outside_count(self, sequence, identities, full):
        if full:
            return 0
        boundary = self._boundaries.get(int(sequence))
        if boundary is None:
            return 0
        return sum(1 for identity, track in boundary["pre"].items()
                   if identity not in identities and track.status != "deleted")

    def _has_cross_component_edge(self, event, replay_tracks, identities):
        boundary = self._boundaries.get(int(event["sequence"]))
        if boundary is None:
            return False
        detections, _ = self._normalize_detections(event["detections"])
        for item in detections:
            for identity, track in boundary["pre"].items():
                if identity in identities or track.status == "deleted":
                    continue
                cost, _ = self._association_cost(track, item["original"], event["timestamp"])
                birth_near = (item["confidence"] >= self.birth_confidence
                              and self._near_confirmed_for_birth(
                                  track, item["bbox"], event["timestamp"],
                                  item["class_id"])[0])
                if cost is not None or birth_near:
                    return True
            for identity, track in replay_tracks.items():
                if identity in identities:
                    continue
                cost, _ = self._association_cost(track, item["original"], event["timestamp"])
                birth_near = (item["confidence"] >= self.birth_confidence
                              and self._near_confirmed_for_birth(
                                  track, item["bbox"], event["timestamp"],
                                  item["class_id"])[0])
                if cost is not None or birth_near:
                    return True
        return False

    def _has_component_edge(self, event, replay_tracks):
        """Detect a new replay-time edge into an event outside the saved component."""
        detections, _ = self._normalize_detections(event["detections"])
        for item in detections:
            for track in replay_tracks.values():
                if track.status == "deleted":
                    continue
                cost, _ = self._association_cost(track, item["original"], event["timestamp"])
                birth_near = (item["confidence"] >= self.birth_confidence
                              and self._near_confirmed_for_birth(
                                  track, item["bbox"], event["timestamp"],
                                  item["class_id"])[0])
                if cost is not None or birth_near:
                    return True
        return False

    def _commit_replay(self, result, source_sequence, new_event, affected_sequences, full):
        identities = result["identities"]
        for identity, retired_at in result.get("retirements", ()):
            self._retired_tracks.setdefault(identity, retired_at)
        for sequence, applied in result["assignments"].items():
            boundary = self._boundaries.get(sequence)
            retired_at = boundary["timestamp"] if boundary is not None else self._latest_timestamp
            for identity in applied.get("retired_ids", ()):
                self._retired_tracks.setdefault(identity, retired_at)
        replay_tracks = self._filter_retired(
            _clone_tracks(result["tracks"]), self._latest_timestamp)
        if full:
            self.tracks = replay_tracks
        else:
            self.tracks = self._merge_snapshot(self.tracks, replay_tracks, identities)
                                                                                       
                                                                                    
            source_pre = self._boundaries[source_sequence]["pre"]
            for identity, track in self.tracks.items():
                if identity not in identities and identity in source_pre:
                    track.misses += 1
        for sequence, snapshot in result["pre"].items():
            boundary = self._boundaries.get(sequence)
            if boundary is None:
                continue
            if full:
                boundary["pre"] = _clone_tracks(snapshot)
            else:
                boundary["pre"] = self._merge_snapshot(boundary["pre"], snapshot, identities)
            self._filter_retired(boundary["pre"], boundary["timestamp"])
        for sequence, snapshot in result["post"].items():
            boundary = self._boundaries.get(sequence)
            if boundary is None:
                continue
            if full:
                boundary["post"] = _clone_tracks(snapshot)
            else:
                boundary["post"] = self._merge_snapshot(boundary["post"], snapshot, identities)
            self._filter_retired(boundary["post"], boundary["timestamp"])
        for sequence, applied in result["assignments"].items():
            for index, event in enumerate(self._events):
                if event["kind"] == "detector" and event["sequence"] == sequence:
                    self._events[index] = applied
                    break
            if sequence == source_sequence:
                new_event = applied
        if not any(event["kind"] == "detector" and event["sequence"] == source_sequence
                   for event in self._events):
            self._events.append(new_event)
        if source_sequence in self._boundaries:
            boundary = self._boundaries[source_sequence]
            if source_sequence not in result["post"]:
                source_post = self._merge_snapshot(boundary["pre"], replay_tracks, identities)
                boundary["post"] = source_post if not full else _clone_tracks(result["tracks"])
                self._filter_retired(boundary["post"], boundary["timestamp"])
        self._filter_retired(self.tracks, self._latest_timestamp)
        self._events.sort(key=self._event_order)
        self._counters["replay_count"] += 1
        if full:
            self._counters["replay_fallback_full"] += 1
        return new_event

    def update_detections(self, detections, source_timestamp, source_sequence,
                          epoch=None, frame_shape=None, active_epoch=None,
                          full_frame=True):
        started = time.perf_counter()
        try:
            timestamp, sequence = float(source_timestamp), int(source_sequence)
        except (TypeError, ValueError):
            self._counters["replay_rejected"] += 1
            return self._reject("invalid_source_time", started)
        if (not math.isfinite(timestamp) or timestamp < 0
                or int(self.stream_epoch if epoch is None else epoch) != self.stream_epoch):
            self._counters["stale_results"] += 1
            return self._reject("stream_epoch_or_timestamp_mismatch", started)
        detections = list(detections or ())
        normalized_for_context, _ = self._normalize_detections(detections)
        detection_classes = {item["class_id"] for item in normalized_for_context}
        oldest_sequence = next(iter(self._boundaries), None)
        oldest_boundary = (self._boundaries.get(oldest_sequence)
                           if oldest_sequence is not None else None)
        source_age_s = (max(0.0, self._latest_timestamp - timestamp)
                        if self._latest_timestamp is not None else None)
        source_age_exceeds_history = (source_age_s is not None
                                      and source_age_s > self.history_seconds + 1e-9)
        source_timestamp_before_oldest = (oldest_boundary is not None
                                          and timestamp < oldest_boundary["timestamp"] - 1e-9)
        source_sequence_before_oldest = (oldest_sequence is not None
                                         and sequence < oldest_sequence)
        source_sequence_ahead = (self._latest_sequence is not None
                                 and sequence > self._latest_sequence)
        source_time_newer_track_ids = sorted(
            str(identity) for identity, track in self.tracks.items()
            if detection_classes and track.class_id in detection_classes
            and track.first_detection_at > timestamp + 1e-9)
        missing_causes = []
        if source_age_exceeds_history:
            missing_causes.append("history_window_too_short")
        if source_timestamp_before_oldest:
            missing_causes.append("detection_older_than_oldest_snapshot")
        if source_sequence_before_oldest:
            missing_causes.append("source_sequence_before_oldest_boundary")
        if source_sequence_ahead:
            missing_causes.append("source_sequence_ahead_of_latest")
        if source_time_newer_track_ids:
            missing_causes.append("track_born_after_source_timestamp")
        if not missing_causes:
            missing_causes.append("source_boundary_not_recorded")
        if source_age_exceeds_history:
            missing_classification = "result_age_exceeds_history_window"
        elif source_timestamp_before_oldest:
            missing_classification = "source_timestamp_older_than_retained_history"
        elif source_sequence_before_oldest:
            missing_classification = "source_sequence_before_retained_history"
        elif source_sequence_ahead:
            missing_classification = "source_sequence_ahead_of_latest"
        else:
            missing_classification = "source_sequence_gap_or_boundary_not_recorded"
        source_context = dict(
            source_sequence=sequence,
            source_timestamp=timestamp,
            source_age_s=source_age_s,
            source_age_exceeds_history_window=source_age_exceeds_history,
            source_timestamp_before_oldest_history=source_timestamp_before_oldest,
            source_before_oldest_boundary=source_sequence_before_oldest,
            source_sequence_ahead_of_latest=source_sequence_ahead,
            source_time_newer_track_ids=list(source_time_newer_track_ids),
            history_oldest_sequence=oldest_sequence,
            history_latest_sequence=self._latest_sequence,
            history_oldest_timestamp=(oldest_boundary["timestamp"]
                                      if oldest_boundary is not None else None),
            history_latest_timestamp=self._latest_timestamp,
            history_window_s=self.history_seconds,
            source_candidate_count=len(detections or ()),
            track_match_attempted=False,
            source_time_track_ids=[])
        boundary = self._boundaries.get(sequence)
        if boundary is None:
            self._counters["stale_results"] += 1
            return self._reject(
                "source_snapshot_missing", started,
                source_snapshot_missing_classification=missing_classification,
                source_snapshot_missing_causes=list(missing_causes),
                **source_context)
        if abs(boundary["timestamp"] - timestamp) > 1e-6:
            self._counters["stale_results"] += 1
            return self._reject("source_frame_timestamp_mismatch", started,
                                source_boundary_timestamp=boundary["timestamp"],
                                **source_context)
        if frame_shape is not None and self._frame_shape is not None and tuple(frame_shape) != self._frame_shape:
            self._counters["stale_results"] += 1
            return self._reject("frame_shape_mismatch", started)
        if any(event["kind"] == "detector" and event["sequence"] == sequence for event in self._events):
            self._counters["duplicate_results"] += 1
            return self._reject("duplicate_source_sequence", started)
        if self._latest_timestamp is None or self._latest_timestamp - timestamp > self.history_seconds + 1e-9:
            self._counters["stale_results"] += 1
            return self._reject("source_result_outside_history", started,
                                **source_context)
        if self._latest_timestamp - timestamp > self.max_result_age + 1e-9:
            self._counters["stale_results"] += 1
            return self._reject("source_result_age_exceeded", started,
                                **source_context)
        source_time_track_ids = set()
        normalized_source_detections, _ = self._normalize_detections(detections)
        for identity, track in boundary["pre"].items():
            retired_at = self._retired_tracks.get(identity)
            if retired_at is not None and retired_at <= timestamp + 1e-9:
                continue
            for item in normalized_source_detections:
                cost, _ = self._association_cost(track, item["original"], timestamp)
                duplicate_guard = (item["confidence"] >= self.birth_confidence
                                  and self._near_confirmed_for_birth(
                                      track, item["bbox"], timestamp,
                                      item["class_id"])[0])
                if cost is not None or duplicate_guard:
                    source_time_track_ids.add(str(identity))
                    break
        new_event = dict(kind="detector", timestamp=timestamp, sequence=sequence,
                         detections=list(detections or ()), epoch=self.stream_epoch,
                         active_epoch=active_epoch, full_frame=bool(full_frame),
                         assignments={}, rejections={},
                         diagnostics={}, applied=False)
        affected_tracks, affected_sequences = self._discover_component(new_event, boundary)
        result, reason, depth, elapsed_ms = self._replay(sequence, new_event,
                                                        affected_tracks, affected_sequences,
                                                        full=False)
        scope = "component"
        if reason == "cross_component_edge":
            all_sequences = {event["sequence"] for event in self._detector_events()}
            all_sequences.add(sequence)
            result, reason, depth, elapsed_ms = self._replay(
                sequence, new_event, set(), all_sequences, full=True)
            scope = "all_tracks_fallback"
        if result is None:
            self._counters["replay_rejected"] += 1
            return self._reject(reason, started, replay_depth=depth,
                                replay_scope=scope, replay_ms=elapsed_ms,
                                affected_tracks=len(affected_tracks))
        replayed_capacity_rejections_added = 0
        for replay_sequence, applied in result.get("assignments", {}).items():
            previous = next((event for event in self._events
                             if event["kind"] == "detector"
                             and event["sequence"] == replay_sequence), None)
            old_keys = self._capacity_rejection_keys_for_event(previous)
            new_keys = self._capacity_rejection_keys_for_event(applied)
                                                                               
                                                                               
            for key in new_keys:
                if int(replay_sequence) == sequence:
                    self._capacity_rejection_keys.add(key)
                elif key not in old_keys and key not in self._capacity_rejection_keys:
                    self._capacity_rejection_keys.add(key)
                    replayed_capacity_rejections_added += 1
        new_event = self._commit_replay(result, sequence, new_event,
                                        affected_sequences if scope == "component" else all_sequences,
                                        scope != "component")
                                                                         
                                                                          
                                                                               
        for applied in result["assignments"].values():
            event_sequence = int(applied["sequence"])
            if applied.get("full_frame", True):
                self._semantic_events[event_sequence] = dict(
                    timestamp=float(applied["timestamp"]),
                    assigned_track_ids=set(applied.get("assignments", {}).values()))
            else:
                self._semantic_events.pop(event_sequence, None)
        semantic_cutoff = self._latest_timestamp - self.semantic_hard_age - self.max_result_age
        self._semantic_events = {
            key: value for key, value in self._semantic_events.items()
            if value["timestamp"] >= semantic_cutoff - 1e-9}
        diag = new_event.get("diagnostics", {})
        self._counters["accepted_detections"] += sum(
            1 for value in new_event.get("assignments", {}).values() if value is not None)
        self._counters["high_score_candidates"] += int(diag.get("high_score_count", 0))
        self._counters["low_score_candidates"] += int(diag.get("low_score_count", 0))
        self._counters["high_score_matches"] += int(diag.get("high_score_matches", 0))
        self._counters["low_score_matches"] += int(diag.get("low_score_matches", 0))
        self._counters["low_score_unmatched"] += int(diag.get("low_score_unmatched", 0))
        self._counters["rejected_detections"] += int(diag.get("invalid_count", 0)) + len(
            new_event.get("rejections", {}))
        self._counters["candidate_columns_limited"] += int(diag.get("limited_count", 0))
        self._counters["tracks_created"] += int(diag.get("tracks_created", 0))
        self._counters["tracks_deleted"] += int(diag.get("tracks_evicted", 0))
        self._last_diagnostic = dict(reason="accepted", source_sequence=sequence,
                                     source_timestamp=timestamp,
                                     result_age_s=max(0.0, self._latest_timestamp - timestamp),
                                     source_age_s=source_age_s,
                                     source_age_exceeds_history_window=source_age_exceeds_history,
                                     source_timestamp_before_oldest_history=source_timestamp_before_oldest,
                                     source_time_newer_track_ids=list(source_time_newer_track_ids),
                                     history_oldest_sequence=oldest_sequence,
                                     history_latest_sequence=self._latest_sequence,
                                     history_oldest_timestamp=(oldest_boundary["timestamp"]
                                                               if oldest_boundary is not None else None),
                                     history_latest_timestamp=self._latest_timestamp,
                                     history_window_s=self.history_seconds,
                                     source_candidate_count=len(detections),
                                     raw_detection_candidates=len(detections or ()),
                                     association_assignments=dict(new_event.get("assignments", {})),
                                     association_costs=dict(diag.get("candidate_costs", {})),
                                     association_detection_summaries=dict(
                                         diag.get("detection_summaries", {})),
                                     association_pairwise_costs=dict(diag.get("pairwise_costs", {})),
                                     association_pairwise_rejections=dict(diag.get("pairwise_rejections", {})),
                                     association_rejections=dict(new_event.get("rejections", {})),
                                     high_score_candidates=diag.get("high_score_count", 0),
                                     low_score_candidates=diag.get("low_score_count", 0),
                                     high_score_matches=diag.get("high_score_matches", 0),
                                     low_score_matches=diag.get("low_score_matches", 0),
                                     low_score_unmatched=diag.get("low_score_unmatched", 0),
                                     invalid_candidates=diag.get("invalid_count", 0),
                                     limited_candidates=diag.get("limited_count", 0),
                                     replay_depth=depth, replay_scope=scope,
                                     replay_ms=elapsed_ms,
                                     affected_track_count=len(affected_tracks),
                                     replayed_capacity_rejections_added=(
                                         replayed_capacity_rejections_added),
                                     tombstone_filtered_track_ids=list(
                                         result.get("tombstone_filtered_track_ids", ())),
                                     track_match_attempted=True,
                                     source_time_track_ids=sorted(source_time_track_ids),
                                     active_epoch=active_epoch)
        self._trim_history()
        return dict(accepted=True, reason="accepted", diagnostics=dict(self._last_diagnostic))

    def _reject(self, reason, started, **details):
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._last_diagnostic = dict(reason=reason, replay_depth=details.pop("replay_depth", 0),
                                     replay_scope=details.pop("replay_scope", "none"),
                                     replay_ms=details.pop("replay_ms", elapsed_ms), **details)
        return dict(accepted=False, reason=reason, diagnostics=dict(self._last_diagnostic))

    @staticmethod
    def _capacity_rejection_keys_for_event(event):
        """Stable identities for replayed capacity rejections in one source frame."""
        if event is None or event.get("kind") != "detector":
            return set()
        summaries = event.get("diagnostics", {}).get("detection_summaries", {})
        keys = set()
        for source_index, reason in event.get("rejections", {}).items():
            if reason != "track_capacity":
                continue
            summary = summaries.get(str(source_index), {})
            identity = summary.get("best_track_id") or summary.get("birth_guard_track_id")
            if identity is None:
                identity = "candidate:%s" % source_index
            else:
                identity = "%s|candidate:%s" % (identity, source_index)
            keys.add((int(event["sequence"]), str(identity), str(reason)))
        return keys

    def _trim_history(self):
        if self._latest_timestamp is None or not self._boundaries:
            return
        cutoff = self._latest_timestamp - self.history_seconds
        sequences = [sequence for sequence, boundary in self._boundaries.items()
                     if boundary["timestamp"] >= cutoff]
        if not sequences:
            sequences = [next(reversed(self._boundaries))]
        first_kept_time = self._boundaries[sequences[0]]["timestamp"]
        keep = set(sequences)
        self._boundaries = OrderedDict((sequence, boundary)
                                       for sequence, boundary in self._boundaries.items()
                                       if sequence in keep)
        self._events = [event for event in self._events
                        if event["timestamp"] >= first_kept_time - 1e-9]
        if len(self._events) > self.max_history_events:
                                                                               
                                          
            self._events = self._events[-self.max_history_events:]
            retained_sequences = {event.get("sequence") for event in self._events
                                  if event["kind"] == "frame"}
            oldest = min(retained_sequences) if retained_sequences else self._latest_sequence
            self._boundaries = OrderedDict((sequence, boundary)
                                           for sequence, boundary in self._boundaries.items()
                                           if sequence >= oldest)
        referenced = set()
        for boundary in self._boundaries.values():
            referenced.update(boundary["pre"])
            referenced.update(boundary["post"])
        tombstone_cutoff = self._latest_timestamp - self.tombstone_retention_seconds
        self._retired_tracks = {
            identity: retired_at for identity, retired_at in self._retired_tracks.items()
            if retired_at >= tombstone_cutoff or identity in referenced
        }
        retained_sequences = set(self._boundaries)
        self._capacity_rejection_keys = {
            key for key in self._capacity_rejection_keys if key[0] in retained_sequences
        }

    def add_lk_velocity(self, observation):
        """Compatibility helper for callers that already advanced to interval end.

        New integrations should pass the event to ``advance`` so it is applied
        before the destination frame's pre-detector boundary snapshot.
        """
        self._counters["lk_velocity_rejected"] += 1
        self._last_diagnostic = dict(reason="lk_must_be_supplied_to_advance_midpoint")
        return False

    def snapshot(self, timestamp=None):
        if self._latest_timestamp is None:
            return []
        timestamp = self._latest_timestamp if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp) or timestamp < self._latest_timestamp - 1e-9:
            return []
        output = []
        detector_events = self._detector_events()
        for identity, current in self.tracks.items():
            track = current.clone()
            track.predict_to(timestamp)
            track.refresh_lifecycle(timestamp, self.lost_after_seconds,
                                    self.retain_seconds, self.tentative_timeout_seconds,
                                    self.confirm_hits)
            if track.status == "deleted":
                continue
            box = _state_box(track.x)
            if self._frame_shape is not None:
                height, width = self._frame_shape[:2]
                box = (max(0.0, box[0]), max(0.0, box[1]),
                       min(float(width), box[2]), min(float(height), box[3]))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            age = max(0.0, timestamp - track.last_detection_at)
            support_at = max(track.last_detection_at,
                             track.last_lk_at if track.last_lk_at is not None
                             else track.last_detection_at)
            support_age = max(0.0, timestamp - support_at)
            semantic_age, full_frame_misses, semantic_state, semantic_reason = (
                self._semantic_evidence(track, timestamp))
            misses = sum(1 for event in detector_events
                         if event["timestamp"] > track.last_detection_at + 1e-9
                         and event["timestamp"] <= timestamp + 1e-9)
            source = "detector" if age <= 1e-9 else "prediction"
            covariance = tuple(float(value) for value in np.diag(track.P)[:4])
            output.append(TrackView(
                track_id=str(identity), bbox=tuple(float(value) for value in box),
                confidence=float(track.last_detection_confidence), class_id=track.class_id,
                status=track.status, hits=track.hits, misses=misses,
                last_detection_at=float(track.last_detection_at),
                last_detection_sequence=int(track.last_detection_sequence),
                source=track.last_source, position_source=source,
                                                                                
                                                                              
                                                       
                valid_until=float(support_at + self.prediction_horizon),
                covariance=covariance, velocity=(float(track.x[4]), float(track.x[5])),
                age_s=age, last_visual_support_at=float(support_at),
                visual_support_age_s=support_age,
                identity_confirmed=track.hits >= self.confirm_hits,
                last_innovation_d2=track.last_innovation_d2,
                last_association_cost=track.last_association_cost,
                rd_source=track.rd_source,
                recent_detection_records=tuple(track.detection_records),
                first_detection_sequence=int(track.first_detection_sequence),
                last_semantic_confirmed_at=float(track.last_semantic_confirmed_at),
                semantic_age_s=semantic_age,
                full_frame_unmatched_count=full_frame_misses,
                semantic_state=semantic_state, semantic_reason=semantic_reason,
                semantic_valid_until=float(track.last_semantic_confirmed_at
                                           + self.semantic_hard_age)))
        output.sort(key=lambda view: (view.track_id != self.active_track_id,
                                      view.status == "lost", view.track_id))
        return output

    def summary(self):
        status_counts = {name: 0 for name in ("tentative", "confirmed", "lost")}
        for track in self.tracks.values():
            if track.status in status_counts:
                status_counts[track.status] += 1
        return dict(stream_epoch=self.stream_epoch, track_count=len(self.tracks),
                    track_status_counts=status_counts, active_track_id=self.active_track_id,
                    identity_confirmed_track_count=sum(
                        track.hits >= self.confirm_hits for track in self.tracks.values()),
                    max_tracks=self.max_tracks,
                    capacity_pool_full=self._capacity_was_full,
                    capacity_log_interval_s=5.0,
                    capacity_rejections_pending=self._capacity_rejections_pending,
                    max_assignment_detections=self.max_assignment_detections,
                    history_seconds=self.history_seconds,
                    history_event_count=len(self._events),
                    oldest_source_sequence=next(iter(self._boundaries), None),
                    latest_source_sequence=self._latest_sequence,
                    prediction_horizon_s=self.prediction_horizon,
                    semantic_soft_age_s=self.semantic_soft_age,
                    semantic_hard_age_s=self.semantic_hard_age,
                    semantic_miss_count=self.semantic_miss_count,
                    semantic_event_count=len(self._semantic_events),
                    detector_anchor_horizon_s=self.detector_anchor_horizon,
                    max_result_age_s=self.max_result_age,
                    high_score_threshold=self.high_score_threshold,
                    low_score_threshold=self.low_score_threshold,
                    birth_confidence=self.birth_confidence,
                    confirm_hits=self.confirm_hits,
                    tentative_timeout_s=self.tentative_timeout_seconds,
                    birth_center_separation_fraction=self.birth_center_separation_fraction,
                    birth_center_max_sigma_fraction=self.birth_center_max_sigma_fraction,
                    tombstone_retention_seconds=self.tombstone_retention_seconds,
                    lost_after_s=self.lost_after_seconds,
                    retain_seconds=self.retain_seconds,
                    retired_track_snapshot_count=len(self._retired_tracks),
                    debug_association_matrix=self.debug_association_matrix,
                    mahalanobis_gate=self.mahalanobis_gate,
                    detector_velocity_guard=dict(innovation_d2=_DETECTOR_VELOCITY_D2,
                                                 center_ratio=_DETECTOR_VELOCITY_CENTER_RATIO,
                                                 speed_short_sides_s=_DETECTOR_VELOCITY_SPEED_RATIO),
                    detector_hard_jump_gate=dict(
                        center_ratio=self.hard_jump_center_ratio,
                        speed_short_sides_s=self.hard_jump_speed_short_sides,
                        applies_after_confirmed=True),
                    replay_budget_ms=self.replay_budget_ms,
                    noise_model=self.noise_model.summary(),
                    last_diagnostic=dict(self._last_diagnostic),
                    **dict(self._counters))
