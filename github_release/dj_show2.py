"""YOLO ONNX + multi-target tracking + single-active-target pan/tilt control.

Run with --help for options. Hardware libraries are loaded only when running.
The default path tracks several candidates but sends one confirmed, time-valid
active track to the pan/tilt controller. Use --legacy-single-target to compare
the previous 3.8 lock path.
"""

import argparse
import csv
import json
import hashlib
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from pan_tilt_control import ServoConfig, open_servo_controller
from latest_frame_capture import FramePacket, LatestFrameCapture
from async_detection import LatestDetectionWorker

VERSION = "4.0.17-lock-priority"


def detector_submission_reason(worker_stats, frame_age_s, max_age_s, prefetch=False):
    """Decide before copying: a queued frame can age while the main loop aligns."""
    if not math.isfinite(frame_age_s) or frame_age_s < 0:
        return 'invalid_frame_age'
    if frame_age_s > max_age_s:
        return 'frame_too_old'
    if not prefetch and worker_stats.get('busy', False):
        return 'worker_busy'
    if not prefetch and worker_stats.get('pending', False):
        return 'worker_pending'
    return 'submitted'


@dataclass(frozen=True)
class Observation:
    bbox: tuple
    confidence: float
    class_id: int
    track_id: object = None
    position_source: str = "detector"
    valid_until: object = None
    motion_reset: bool = False


def remap_roi_detections(detections, roi):
    """Return detector boxes in full source-frame coordinates."""
    left, top = roi[:2]
    return [replace(item, bbox=(item.bbox[0] + left, item.bbox[1] + top,
                                item.bbox[2] + left, item.bbox[3] + top))
            for item in detections]


def mot_search_needed(selection):
    target = None if selection is None else selection.target
    return (target is None or
            getattr(target, "semantic_state", "verified") == "suspect")


def mot_search_roi_eligible(view):
    return (getattr(view, "semantic_state", "verified") != "invalid"
            and view.visual_support_age_s <= .7
            and view.status in ("tentative", "confirmed", "lost"))


def bbox_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def attach_track_ids(detections, tracks, minimum_iou=0.1):
    """Associate returned tracks to detections instead of reusing list indices."""
    result = list(detections)
    pairs = sorted(
        ((bbox_iou(det.bbox, track[:4]), i, j)
         for i, det in enumerate(detections) for j, track in enumerate(tracks)),
        reverse=True,
    )
    used_detections, used_tracks = set(), set()
    for overlap, i, j in pairs:
        if overlap < minimum_iou or i in used_detections or j in used_tracks:
            continue
        result[i] = replace(detections[i], track_id=int(tracks[j][4]))
        used_detections.add(i)
        used_tracks.add(j)
    return result


class SceneMetrics:
    """Measure one physical target; suppress identity claims on ambiguous frames."""

    def __init__(self, deadzone=0.05, settle_frames=3, lost_frames=3):
        if not 0 <= deadzone < 1 or settle_frames < 1 or lost_frames < 1:
            raise ValueError("Invalid deadzone, settle_frames or lost_frames")
        self.deadzone = deadzone
        self.settle_frames = settle_frames
        self.lost_frames = lost_frames
        self.frames = self.visible_frames = self.ambiguous_frames = 0
        self.loss_count = self.id_switch_count = self.missing_streak = 0
        self.last_track_id = None
        self.seen_target = self.loss_active = False
        self.center_started_at = None
        self.center_initially_inside = self.centered = False
        self.center_streak = self.centering_events = 0
        self.last_center_time_s = None
        self.center_time_sum = self.error_sum = self.error_squared_sum = 0.0
        self.last_state = {}

    def observe(self, detections, width, height, timestamp):
        if width <= 0 or height <= 0:
            raise ValueError("Image size must be positive")
        self.frames += 1
        state = {
            "status": "missing", "target_count": len(detections), "track_id": None,
            "error_x": None, "error_y": None, "error_norm": None,
            "in_deadzone": None, "center_confirmed": False,
            "centering_elapsed_s": None,
        }
        if len(detections) > 1:
            self.ambiguous_frames += 1
            state["status"] = "ambiguous"
                                                                                    
            self.last_track_id = self.center_started_at = None
            self.seen_target = self.loss_active = self.centered = False
            self.missing_streak = self.center_streak = 0
        elif not detections:
            self.center_streak = 0
            self.centered = False
            if self.seen_target:
                self.missing_streak += 1
                if self.missing_streak >= self.lost_frames and not self.loss_active:
                    self.loss_count += 1
                    self.loss_active = True
        else:
            target = detections[0]
            self.seen_target = True
            self.visible_frames += 1
            self.missing_streak = 0
            self.loss_active = False
            if target.track_id is not None:
                if self.last_track_id is not None and target.track_id != self.last_track_id:
                    self.id_switch_count += 1
                self.last_track_id = target.track_id
            x1, y1, x2, y2 = target.bbox
            ex = ((x1 + x2) / 2.0 - width / 2.0) / (width / 2.0)
            ey = ((y1 + y2) / 2.0 - height / 2.0) / (height / 2.0)
            error = math.hypot(ex, ey)
            inside = abs(ex) <= self.deadzone and abs(ey) <= self.deadzone
            self.error_sum += error
            self.error_squared_sum += error * error
            if self.center_started_at is None and not self.centered:
                self.center_started_at = timestamp
                self.center_initially_inside = inside
            if not inside:
                if self.centered:
                    self.center_started_at = timestamp
                self.center_initially_inside = self.centered = False
                self.center_streak = 0
            else:
                self.center_streak += 1
                if self.center_streak >= self.settle_frames and not self.centered:
                    self.last_center_time_s = (
                        0.0 if self.center_initially_inside
                        else timestamp - self.center_started_at
                    )
                    self.center_time_sum += self.last_center_time_s
                    self.centering_events += 1
                    self.centered = True
                    self.center_started_at = None
            state.update(
                status="visible", track_id=target.track_id,
                error_x=ex, error_y=ey, error_norm=error,
                in_deadzone=inside, center_confirmed=self.centered,
            )
        if self.center_started_at is not None:
            state["centering_elapsed_s"] = max(0.0, timestamp - self.center_started_at)
        state.update(
            loss_count=self.loss_count, missing_streak=self.missing_streak,
            id_switch_count=self.id_switch_count, ambiguous_frames=self.ambiguous_frames,
            last_center_time_s=self.last_center_time_s, centering_events=self.centering_events,
        )
        self.last_state = state
        return state

    def summary(self):
        return {
            "measurement_frames": self.frames,
            "single_target_frames": self.visible_frames,
            "ambiguous_frames": self.ambiguous_frames,
            "loss_count": self.loss_count,
            "id_switch_count": self.id_switch_count,
            "centering_events": self.centering_events,
            "last_center_time_s": self.last_center_time_s,
            "mean_center_time_s": (
                self.center_time_sum / self.centering_events if self.centering_events else None
            ),
            "centering_pending": self.center_started_at is not None,
            "mean_error_norm": self.error_sum / self.visible_frames if self.visible_frames else None,
            "rms_error_norm": (
                math.sqrt(self.error_squared_sum / self.visible_frames) if self.visible_frames else None
            ),
        }


class FrameMetrics:
    def __init__(self, started_at, window_size=30):
        if window_size < 2:
            raise ValueError("FPS window must contain at least two frames")
        self.started_at = started_at
        self.completed_at = None
        self.completions = deque(maxlen=window_size)
        self.frame_count = 0
        self.inference_sum = self.frame_sum = 0.0
        self.inference_count = 0
        self.last_state = {"frame_ms": None, "actual_fps": None}

    def record(self, started_at, completed_at, inference_ms):
        self.frame_count += 1
        self.completed_at = completed_at
        self.completions.append(completed_at)
        frame_ms = (completed_at - started_at) * 1000.0
        if inference_ms is not None:
            self.inference_sum += inference_ms
            self.inference_count += 1
        self.frame_sum += frame_ms
        span = self.completions[-1] - self.completions[0]
        elapsed = completed_at - self.started_at
        self.last_state = {
            "frame_index": self.frame_count,
            "elapsed_s": elapsed,
            "inference_ms": inference_ms,
            "frame_ms": frame_ms,
            "actual_fps": (len(self.completions) - 1) / span if span > 0 else None,
            "average_fps": self.frame_count / elapsed if elapsed > 0 else None,
        }
        return self.last_state

    def summary(self):
        elapsed = self.completed_at - self.started_at if self.completed_at is not None else 0.0
        return {
            "processed_frames": self.frame_count,
            "elapsed_s": elapsed,
            "average_fps": self.frame_count / elapsed if elapsed > 0 else None,
            "inference_frames": self.inference_count,
            "mean_inference_ms": self.inference_sum / self.inference_count if self.inference_count else None,
            "mean_frame_ms": self.frame_sum / self.frame_count if self.frame_count else None,
        }


class CsvLog:
    fields = (
        "wall_time", "frame_index", "elapsed_s", "inference_ms", "frame_ms",
        "actual_fps", "average_fps", "status", "target_count", "track_id",
        "error_x", "error_y", "error_norm", "in_deadzone", "center_confirmed",
        "centering_elapsed_s", "last_center_time_s", "centering_events",
        "missing_streak", "loss_count", "id_switch_count", "ambiguous_frames",
        "capture_ms", "vision_other_ms", "control_ms", "control_decision_delay_ms", "output_ms",
        "control_status", "control_position_valid", "control_track_id",
        "control_timestamp", "control_selected_track_id", "control_bound_track_id",
        "control_visual_support_at", "control_visual_support_age_s", "control_valid_until",
        "control_reject_reason",
        "pan_angle_deg", "tilt_angle_deg",
        "pan_step_deg", "tilt_step_deg", "pan_limited", "tilt_limited",
        "filtered_error_x", "filtered_error_y",
        "pan_reason", "tilt_reason", "pan_at_limit", "tilt_at_limit",
        "frame_sequence", "frames_overwritten", "source_read_ms",
        "seq_delta", "video_frame_index",
        "frame_age_ms", "display_frame_age_ms",
        "control_bbox", "control_confidence", "control_error_x", "control_error_y",
        "pan_pid_p", "pan_pid_i", "pan_pid_d", "pan_pid_output_deg_s",
        "tilt_pid_p", "tilt_pid_i", "tilt_pid_d", "tilt_pid_output_deg_s",
        "pan_rate_deg_s", "tilt_rate_deg_s", "pan_pid_saturated", "tilt_pid_saturated",
        "pan_feedforward_deg_s", "tilt_feedforward_deg_s",
        "pan_command_output_deg_s", "tilt_command_output_deg_s",
        "target_pan_velocity_deg_s", "target_tilt_velocity_deg_s", "velocity_ready", "velocity_samples",
        "detector_ran", "raw_target_detections", "lock_status", "position_source", "visual_backend",
        "position_age_s", "detection_age_s", "detector_track_id", "control_position_source",
        "detector_submitted", "detector_busy", "detector_accepted", "detector_result_age_ms",
        "detector_search_roi", "detector_full_frame",
        "last_semantic_confirmed_at", "semantic_age_s", "semantic_state",
        "semantic_reason", "full_frame_unmatched_count", "semantic_transition",
        "association_reason", "association_stage", "mot_high_score_candidates", "mot_low_score_candidates",
        "mot_high_score_matches", "mot_low_score_matches", "mot_low_score_unmatched",
        "association_candidates", "identity_recovered", "position_alignment",
        "visual_update_ok", "detector_source_sequence",
        "recovery_pending_count",
        "visual_failure_reason", "visual_point_count",
        "lk_fwd_ok", "lk_bwd_ok", "lk_roundtrip_ok", "lk_visual_used", "lk_ransac_reject_reason",
        "lk_reject_reason", "lk_gate_rho", "lk_offset_ratio", "lk_center_ratio",
        "lk_velocity_ratio", "lk_r_scale", "lk_centroid_offset_ratio",
        "lk_center_innovation_ratio", "lk_velocity_discrepancy_ratio",
        "visual_tracking_ms", "visual_init_ms", "alignment_ms", "history_store_ms",
        "visual_inlier_count", "visual_inlier_ratio", "visual_reprojection_error",
        "visual_fb_error", "visual_scale", "visual_rotation_deg", "visual_roi_area",
        "alignment_steps", "alignment_reason", "alignment_method", "alignment_budget_exceeded",
        "detector_poll_ms", "detector_submit_ms", "overlay_ms", "video_write_ms", "display_ms",
        "detector_submit_reason", "detector_request_age_ms",
        "detector_reanchor_pending_count", "detector_reanchored",
        "fusion_uncertainty_px", "fusion_observed_at", "fusion_provenance",
        "template_ready", "template_verified", "template_score", "template_peak_margin",
        "template_search_ms", "template_status", "template_confirmations",
        "template_seed_reject_reason", "template_anchor_score", "template_best_score",
        "template_peak_margin_diagnostic", "template_timeout", "template_support_expired",
        "flow_motion_hint_used", "flow_guided_points", "raw_detection_candidates",
        "active_track_id", "active_epoch", "track_count", "track_summaries",
        "active_frontend_binding",
        "position_valid", "position_invalid_reason", "manual_active_lock",
        "selector_candidate_diagnostics",
        "visual_support_age_s",
        "association_assignments", "association_costs", "association_rejections",
        "association_detection_summaries", "association_pairwise_costs",
        "association_pairwise_rejections", "pending_switch_id", "pending_switch_confirmations",
        "ambiguous_duration_s",
        "manual_acquire_result", "track_pool_full", "track_capacity_event",
        "track_capacity_rejections",
        "replay_depth", "replay_scope", "replay_ms", "replay_reason",
        "replay_source_sequence", "replay_source_timestamp",
        "replay_source_age_s", "replay_source_before_oldest_boundary",
        "replay_source_sequence_ahead_of_latest",
        "replay_source_age_exceeds_history_window",
        "replay_source_timestamp_before_oldest_history",
        "replay_source_snapshot_missing_classification",
        "replay_source_snapshot_missing_causes",
        "replay_source_time_newer_track_ids",
        "replay_source_boundary_timestamp", "replay_history_oldest_sequence",
        "replay_history_latest_sequence", "replay_history_oldest_timestamp",
        "replay_history_latest_timestamp", "replay_history_window_s",
        "replay_source_candidate_count", "replay_track_match_attempted",
        "replay_source_time_track_ids", "replay_tombstone_track_ids",
        "candidate_limited_count", "kf_covariance", "stream_epoch",
    )

    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields, extrasaction="raise")
        self.writer.writeheader()
        return self

    def write(self, values):
        self.writer.writerow(values)
        self.file.flush()

    def __exit__(self, *args):
        if self.file is not None:
            self.file.close()


class yolov5_lite:
    """ONNX detector with optional legacy SORT; MOT owns IDs in the default path."""

    def __init__(self, modelpath, classfile, confidence=0.35, detector_threads=1, assign_ids=False, ort_spinning=True):
        import cv2
        import numpy as np
        import onnxruntime as ort

        self.cv2, self.np = cv2, np
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = detector_threads
        options.inter_op_num_threads = 1
        self.ort_spinning_applied = callable(getattr(options, 'add_session_config_entry', None))
        if self.ort_spinning_applied:
            for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
                options.add_session_config_entry(key, '1' if ort_spinning else '0')
        self.net = ort.InferenceSession(str(modelpath), options)
        with Path(classfile).open(encoding="utf-8-sig") as labels:
            self.classes = [line.strip() for line in labels]
        self.confidence = confidence
        self.input_name = self.net.get_inputs()[0].name
        shape = self.net.get_inputs()[0].shape
        if len(shape) != 4 or shape[1] != 3 or not all(
            isinstance(value, int) and value > 0 for value in shape[2:]
        ):
            raise ValueError("Expected a fixed-size NCHW RGB ONNX input, got %r" % (shape,))
        self.input_h, self.input_w = shape[2:]
        self.tracker = None
        if assign_ids:
            self.enable_sort()

    def enable_sort(self):
        if self.tracker is None:
            from sort.sort import Sort
            self.tracker = Sort()

    def letterBox(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.input_w / width, self.input_h / height)
        neww = max(1, min(self.input_w, round(width * scale)))
        newh = max(1, min(self.input_h, round(height * scale)))
        left = (self.input_w - neww) // 2
        top = (self.input_h - newh) // 2
        resized = self.cv2.resize(frame, (neww, newh), interpolation=self.cv2.INTER_AREA)
        image = self.cv2.copyMakeBorder(
            resized, top, self.input_h - newh - top, left,
            self.input_w - neww - left, self.cv2.BORDER_CONSTANT, value=0,
        )
        return image, newh, neww, top, left

    def detect(self, frame):
        cv2, np = self.cv2, self.np
        image, newh, neww, top, left = self.letterBox(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(image.transpose(2, 0, 1), 0))
        started = time.perf_counter()
        output = self.net.run(None, {self.input_name: blob})[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        output = np.asarray(output)
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2 or output.shape[1] != 6:
            raise ValueError(
                "Expected post-NMS rows [x1,y1,x2,y2,score,class_id], got %r. "
                "Check ONNX export/output format." % (output.shape,)
            )
        height, width = frame.shape[:2]
        detections = []
        for row in output:
            if not np.isfinite(row).all() or row[4] <= self.confidence:
                continue
            x1 = max(0.0, min(float(width), (float(row[0]) - left) * width / neww))
            y1 = max(0.0, min(float(height), (float(row[1]) - top) * height / newh))
            x2 = max(0.0, min(float(width), (float(row[2]) - left) * width / neww))
            y2 = max(0.0, min(float(height), (float(row[3]) - top) * height / newh))
            if x2 > x1 and y2 > y1:
                detections.append(Observation((x1, y1, x2, y2), float(row[4]), int(row[5])))
        if self.tracker is None:
            return detections, [], inference_ms
        rows = np.asarray([list(det.bbox) + [det.confidence] for det in detections], dtype=np.float32)
        tracks = self.tracker.update(rows.reshape(-1, 5))
        return attach_track_ids(detections, tracks), tracks, inference_ms

    def draw(self, frame, detections, control=None):
        selected_bbox = control.get("control_bbox") if control else None
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            name = self.classes[det.class_id] if 0 <= det.class_id < len(self.classes) else "Unknown"
            identity = str(det.track_id) if det.track_id is not None else "pending"
            if getattr(det, "semantic_state", "verified") == "invalid":
                color = (0, 165, 255)
                for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
                    segments = max(1, math.ceil(math.dist(a, b) / 12))
                    for index in range(0, segments, 2):
                        start = tuple(round(a[k] + (b[k] - a[k]) * index / segments)
                                      for k in (0, 1))
                        end = tuple(round(a[k] + (b[k] - a[k])
                                          * min(index + 1, segments) / segments)
                                    for k in (0, 1))
                        self.cv2.line(frame, start, end, color, 1)
                self.cv2.putText(frame, "%s ID:%s UNVERIFIED" % (name, identity),
                                 (x1, max(15, y1 - 8)), self.cv2.FONT_HERSHEY_SIMPLEX,
                                 0.5, color, 1, self.cv2.LINE_AA)
                continue
            selected = selected_bbox is not None and list(det.bbox) == selected_bbox
            lifecycle = getattr(det, "status", "")
            source = getattr(det, "position_source", "detector")
            color = (255, 255, 0) if selected else (
                (0, 165, 255) if lifecycle == "lost" or source == "prediction"
                else (0, 200, 0) if lifecycle == "confirmed"
                else (160, 160, 160))
            score_label = "%.2f" % det.confidence if source == "detector" else source.upper()
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 2)
            self.cv2.putText(
                frame, "%s%s ID:%s %s %s" % ("CONTROL " if selected else "", name,
                                                identity, lifecycle.upper() or "DETECT", score_label),
                (x1, max(15, y1 - 8)), self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1,
            )


def number(value, digits=2):
    return "N/A" if value is None else ("%.*f" % (digits, value))


def overlay(cv2, frame, scene, previous_timing, inference_ms, deadzone, control=None, show_stats=True, tracking_note=None):
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    half_x, half_y = round(width * deadzone / 2), round(height * deadzone / 2)
    lines = [
        "Infer:%s ms | Prev frame:%s ms | FPS:%s" % (
            number(inference_ms), number(previous_timing.get("frame_ms")),
            number(previous_timing.get("actual_fps"))),
        "Target:%s | ID:%s | candidates:%s" % (
            scene["status"], scene["track_id"], scene["target_count"]),
        "Error x:%s y:%s norm:%s" % (
            number(scene["error_x"], 3), number(scene["error_y"], 3), number(scene["error_norm"], 3)),
        "In zone:%s | settled:%s | last center:%s s" % (
            scene["in_deadzone"], scene["center_confirmed"], number(scene["last_center_time_s"], 3)),
        "Lost:%d | ID changes:%d | ambiguous frames:%d" % (
            scene["loss_count"], scene["id_switch_count"], scene["ambiguous_frames"]),
    ]
    if control is not None:
        lines.append("PID:%s | target valid:%s | pan:%s tilt:%s | limit:%s/%s" % (
            control["control_status"], control.get("control_position_valid", False),
            number(control.get("pan_angle_deg"), 2),
            number(control.get("tilt_angle_deg"), 2), control.get("pan_limited", False),
            control.get("tilt_limited", False)))
        lines.append("Pan:%s | tilt:%s | prev age:%s ms" % (
            control.get("pan_reason", "N/A"), control.get("tilt_reason", "N/A"),
            number(previous_timing.get("frame_age_ms"))))
        lines.append("Control ex:%s ey:%s | step:%s/%s deg" % (
            number(control.get("control_error_x"), 3), number(control.get("control_error_y"), 3),
            number(control.get("pan_step_deg"), 3), number(control.get("tilt_step_deg"), 3)))
    scale = max(0.25, min(0.5, width / 1300))
    line_height = max(14, round(scale * 38))
    if show_stats:
        cv2.rectangle(frame, (0, 0), (width - 1, min(height - 1, line_height * len(lines) + 8)), (25, 25, 25), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (4, (i + 1) * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    if tracking_note:
        cv2.putText(frame, tracking_note, (6, max(16, height - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                                                                            
    cv2.rectangle(frame, (center_x - half_x, center_y - half_y),
                  (center_x + half_x, center_y + half_y), (0, 255, 255), 1)
    cv2.drawMarker(frame, (center_x, center_y), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)


def fit_display(cv2, frame, max_width, max_height):
    """Fit the whole source image in a view box without changing control coordinates."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 1.0:
        return frame
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def mot_click_callback(event, x, y, flags, state):
    """Queue a click on a confirmed track; tracker state remains main-thread-owned."""
    if event != state.get("left_button_event"):
        return
    scale_x = float(state.get("scale_x", 1.0))
    scale_y = float(state.get("scale_y", 1.0))
    if scale_x <= 0 or scale_y <= 0:
        return
    source_x, source_y = float(x) / scale_x, float(y) / scale_y
    hits = []
    for track in state.get("tracks", ()):
        if (getattr(track, "status", "") not in ("confirmed", "lost")
                or not getattr(track, "identity_confirmed", False)):
            continue
        x1, y1, x2, y2 = map(float, track.bbox)
        if x1 <= source_x <= x2 and y1 <= source_y <= y2:
            center_x, center_y = (x1 + x2) * .5, (y1 + y2) * .5
            area = max(1.0, (x2 - x1) * (y2 - y1))
            distance2 = (source_x - center_x) ** 2 + (source_y - center_y) ** 2
            hits.append((area, distance2, str(track.track_id)))
    if hits:
        hits.sort()
        state["request_track_id"] = hits[0][2]
        state["click_feedback"] = "queued:%s" % hits[0][2]
    else:
        state["click_feedback"] = "no_confirmed_track_under_click"


def build_parser():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelpath", type=Path, default=root / "models" / "best.onnx")
    parser.add_argument("--classfile", type=Path, default=root / "classes.names")
    parser.add_argument("--source", default="0", help="Camera number or video path")
    parser.add_argument("--confThreshold", type=float, default=0.35)
    parser.add_argument("--target-class", type=int, default=0, help="Class ID to measure; default: 0 (UAV)")
    parser.add_argument("--deadzone", type=float, default=0.05, help="Half-image-normalized tolerance on EACH axis")
    parser.add_argument("--settle-frames", type=int, default=3)
    parser.add_argument("--lost-frames", type=int, default=3)
    parser.add_argument("--fps-window", type=int, default=30)
    parser.add_argument("--no-servo", action="store_true", help="Do not import or initialize servo hardware")
    parser.add_argument("--deadzone-control", action="store_true", help="Compatibility option: deadzone control is now always enabled")
    parser.add_argument("--pan-channel", type=int, default=1)
    parser.add_argument("--tilt-channel", type=int, default=0)
    parser.add_argument("--pan-home", type=float, default=135.0)
    parser.add_argument("--tilt-home", type=float, default=90.0)
    parser.add_argument("--pan-min", type=float, default=0.0)
    parser.add_argument("--pan-max", type=float, default=270.0)
    parser.add_argument("--tilt-min", type=float, default=0.0)
    parser.add_argument("--tilt-max", type=float, default=270.0)
    parser.add_argument("--pan-direction", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--tilt-direction", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--pan-gain", "--pan-kp", dest="pan_gain", type=float, default=40.0, help="Horizontal PID Kp; degrees/second per normalized error")
    parser.add_argument("--tilt-gain", "--tilt-kp", dest="tilt_gain", type=float, default=40.0)
    parser.add_argument("--pan-ki", type=float, default=2.0)
    parser.add_argument("--tilt-ki", type=float, default=2.0)
    parser.add_argument("--pan-kd", type=float, default=1.0)
    parser.add_argument("--tilt-kd", type=float, default=1.0)
    parser.add_argument("--pid-integral-limit", type=float, default=0.5, help="Absolute bound on integral of normalized error in seconds")
    parser.add_argument("--pid-derivative-smoothing", type=float, default=0.25, help="Newest derivative weight in (0,1]")
    parser.add_argument("--pid-reset-gap", type=float, default=0.5, help="Clear PID memory after this many seconds between updates")
    parser.add_argument("--velocity-feedforward", type=float, default=0.5, help="Angular velocity feedforward gain; default: 0.5; 0 disables it")
    parser.add_argument("--horizontal-fov", type=float, default=62.0, help="Actual stream horizontal FOV in degrees; nominal, must be calibrated")
    parser.add_argument("--velocity-smoothing", type=float, default=0.3, help="Newest angular-rate estimate weight in (0,1]")
    parser.add_argument("--feedforward-max-speed", type=float, default=15.0, help="Independent feedforward rate cap in degrees/second")
    parser.add_argument("--feedforward-deadband", type=float, default=1.0, help="Suppress feedforward below this rate in degrees/second")
    parser.add_argument("--servo-max-speed", type=float, default=30.0, help="Maximum commanded degrees/second per axis")
    parser.add_argument("--servo-max-step", type=float, default=5.0, help="Maximum commanded degrees per processed frame")
    parser.add_argument("--servo-smoothing", type=float, default=0.5, help="EMA weight for the newest observed center")
    parser.add_argument("--target-hold", type=float, default=1.0, help="Seconds before releasing a lost target lock")
    parser.add_argument("--target-gate", type=float, default=0.35, help="Normalized center displacement allowed when SORT ID is unavailable")
    parser.add_argument("--no-visual-lock", action="store_true", help="Use the original detection-only position pipeline")
    parser.add_argument("--legacy-single-target", action="store_true",
                        help="Run the 3.8 single-target lock path for comparison; default is multi-target tracking")
    parser.add_argument("--legacy-fusion", action="store_true", help="Comparison: restore 3.6.3 affine-box fusion and image-based delayed alignment")
    parser.add_argument("--no-template-recovery", action="store_true", help="Comparison: disable protected appearance templates and keep 3.7 source-aware fusion")
    parser.add_argument("--visual-tracker", choices=("auto", "kcf", "flow", "flow-legacy"), default="flow")
    parser.add_argument("--flow-roi-scale", type=float, default=3.0)
    parser.add_argument("--flow-min-inliers", type=int, default=6)
    parser.add_argument("--flow-min-inlier-ratio", type=float, default=.55)
    parser.add_argument("--alignment-max-steps", type=int, default=3, help="Legacy fusion image-projection steps")
    parser.add_argument("--alignment-budget-ms", type=float, default=200.0, help="Legacy fusion image-projection soft budget")
    parser.add_argument("--ort-spinning", action=argparse.BooleanOptionalAction, default=True,
                        help="ORT thread spinning; compare --no-ort-spinning on the Pi")
    parser.add_argument("--detect-interval", type=float, default=0.25, help="Seconds between YOLO corrections while visual tracking succeeds")
    parser.add_argument("--lock-prediction", type=float, default=0.35, help="Maximum seconds of unobserved motion prediction; separate from identity retention")
    parser.add_argument("--lock-release", type=float, default=1.0, help="Release target after this many seconds without image position")
    detection_modes = parser.add_mutually_exclusive_group()
    detection_modes.add_argument("--sync-detect", action="store_true", help="Run YOLO synchronously for comparison")
    detection_modes.add_argument("--async-detect", action="store_true", help="Also use asynchronous detection for video files (live cameras default to async)")
    parser.add_argument("--detector-threads", type=int, default=1, help="ONNX intra-op threads; default 1 based on this Pi's tests; 0 restores runtime automatic selection")
    parser.add_argument("--detector-prefetch", action="store_true", help="Comparison: allow one replaceable pending detector frame; default submits only when idle")
    parser.add_argument("--opencv-threads", type=int, default=1, help="OpenCV internal threads; limit CPU competition with YOLO")
    parser.add_argument("--max-detection-age", type=float, default=0.60, help="Reject older async detection images; accepted boxes still require image alignment")
    parser.add_argument("--mot-max-tracks", type=int, default=8, help="Maximum simultaneous MOT tracks")
    parser.add_argument("--mot-max-detections", type=int, default=16, help="Maximum YOLO candidates sent to one assignment step")
    parser.add_argument("--mot-history-seconds", type=float, default=.5, help="Bounded source-time replay history in seconds")
    parser.add_argument("--mot-search-interval", type=float, default=.15,
                        help="YOLO interval while no active target is currently control-valid")
    parser.add_argument("--mot-confirm-hits", type=int, default=2, help="Distinct source-frame detections required to confirm a track")
    parser.add_argument("--mot-birth-confidence", type=float, default=.15,
                        help="Minimum score to create a tentative track; confirmation still requires distinct detector frames")
    parser.add_argument("--mot-low-score-threshold", type=float, default=.15,
                        help="Lowest detector score retained for second-stage association and tentative track births")
    parser.add_argument("--mot-birth-center-separation-fraction", type=float, default=.5,
                        help="Minimum candidate-to-confirmed-track center distance as a fraction of the confirmed box short side")
    parser.add_argument("--mot-birth-center-max-sigma-fraction", type=float, default=.5,
                        help="Apply duplicate-birth center guard only if predicted position sigma is at most this fraction of the confirmed box short side")
    parser.add_argument("--mot-tombstone-retention-seconds", type=float, default=1.0,
                        help="Keep retired-ID tombstones for at least the replay history window")
    parser.add_argument("--mot-lost-after", type=float, default=.25,
                        help="Provisional seconds without YOLO before lost; calibrate to >=1.5x detection-interval P90 and keep below prediction horizon")
    parser.add_argument("--mot-tentative-timeout", type=float, default=1.0,
                        help="Seconds allowed for an unconfirmed tentative track to receive a second detection")
    parser.add_argument("--mot-retain-seconds", type=float, default=3.0, help="Seconds to retain lost tracks for detector reassociation")
    parser.add_argument("--mot-semantic-soft-age", type=float, default=1.2,
                        help="Seconds after the last matched YOLO source frame before full-frame misses mark an active track suspect")
    parser.add_argument("--mot-semantic-hard-age", type=float, default=2.0,
                        help="Maximum seconds of LK-only identity support before control becomes invalid")
    parser.add_argument("--mot-semantic-misses", type=int, default=3,
                        help="Distinct accepted full-frame YOLO misses needed to mark an active track suspect")
    parser.add_argument("--mot-prediction-horizon", type=float, default=.35, help="Maximum age of YOLO or qualified LK support for active control output")
    parser.add_argument("--mot-detector-anchor-horizon", type=float, default=.75,
                        help="Legacy diagnostic interval for detector anchors; fresh qualified LK can extend active support")
    parser.add_argument("--mot-show-all-tracks", action="store_true",
                        help="Debug display of all retained MOT boxes, including stale predictions")
    parser.add_argument("--mot-debug-detections", action="store_true",
                        help="Save source-aligned detector frames with every low/high-score candidate and association result")
    parser.add_argument("--mot-click-select", action="store_true",
                        help="Show MOT tracks and click a confirmed/lost track to manually lock that ID")
    parser.add_argument("--mot-active-hold-seconds", type=float, default=.35, help="Maximum short-term position prediction after support is lost; active ID remains retained until mot-retain-seconds")
    parser.add_argument("--mot-switch-confidence", type=float, default=.15, help="Minimum candidate score for confirmed automatic failover; requires distinct-frame track confirmation")
    parser.add_argument("--mot-switch-confirmations", type=int, default=2, help="Compatibility option; track confirmation already requires distinct detector source frames")
    parser.add_argument("--mot-switch-window", type=float, default=1.25, help="Maximum time window between active-switch confirmations")
    parser.add_argument("--mot-mahalanobis-gate", type=float, default=18.47, help="Four-dimensional detector innovation gate")
    parser.add_argument("--mot-replay-budget-ms", type=float, default=12.0, help="Maximum source-time replay budget; over-budget late packets are rejected")
    parser.add_argument("--mot-jump-center-ratio", type=float, default=2.0,
                        help="Reject a confirmed-track association when detector center jump reaches this many target short sides")
    parser.add_argument("--mot-jump-speed-short-sides", type=float, default=6.0,
                        help="Jump gate minimum detector-anchor speed in target short sides per second")
    parser.add_argument("--mot-debug-association-matrix", action="store_true",
                        help="Log full per-detection/per-track association costs and gate rejections")
    parser.add_argument("--mot-rd-calibration", type=Path, default=None,
                        help="Optional labeled-data YOLO covariance bucket JSON; absent buckets use logged fallback R_D")
    parser.add_argument("--lk-centroid-soft-limit", type=float, default=1.0,
                        help="Initial uncalibrated inlier-centroid offset limit in target short-side units")
    parser.add_argument("--lk-center-soft-limit", type=float, default=.75,
                        help="Initial uncalibrated LK/KF center-innovation limit in target short-side units")
    parser.add_argument("--lk-velocity-soft-limit", type=float, default=1.5,
                        help="Initial uncalibrated LK/KF velocity residual limit")
    parser.add_argument("--lk-velocity-sigma", type=float, default=18.0,
                        help="Base uncalibrated LK speed-observation standard deviation in px/s")
    identity_modes = parser.add_mutually_exclusive_group()
    identity_modes.add_argument("--single-target-recovery", action="store_true", default=True, help="Default: unique-target motion continuity; assumes one physical target, not ReID")
    identity_modes.add_argument("--strict-identity", dest="single_target_recovery", action="store_false", help="Require strong appearance evidence to recover identity")
    identity_modes.add_argument("--no-identity-memory", action="store_true", help="Disable short-term identity recovery")
    parser.add_argument("--recovery-timeout", type=float, default=3.0, help="Identity memory in seconds without a measured image position; does not extend blind control")
    parser.add_argument("--recovery-confirmations", type=int, default=2, help="Consecutive unambiguous appearance matches required for recovery")
    parser.add_argument("--recovery-gate", type=float, default=.65, help="Maximum normalized weak-candidate recovery displacement; legacy grows with measurement age")
    parser.add_argument("--recovery-confirm-window", type=float, default=1.25, help="Maximum seconds between consistent recovery confirmations")
    parser.add_argument("--recovery-area-ratio", type=float, default=4., help="Weak-candidate recovery area bound and reciprocal; strong confirmed recovery can replace stale scale")
    parser.add_argument("--no-detector-reanchor", action="store_true", help="Disable confirmed high-confidence detector correction of a drifting visual lock")
    parser.add_argument("--detector-reanchor-after", type=float, default=.75, help="Required age of the last supporting detector result before reanchor proposals")
    parser.add_argument("--detector-reanchor-confidence", type=float, default=.5, help="Independent strong-detection threshold; two distinct consistent source frames required")
    parser.add_argument("--appearance-threshold", type=float, default=0.78, help="Minimum appearance similarity for recovery; requires real-image calibration")
    parser.add_argument("--home-wait", type=float, default=1.0, help="Seconds to wait after commanding the initial pose")
    parser.add_argument("--home-only", action="store_true", help="Command initial pose and exit without loading model/camera")
    parser.add_argument("--servo-test", choices=("pan", "tilt"), default=None,
                        help="Probe ONLY the selected axis around home, without camera/model")
    parser.add_argument("--servo-test-step", type=float, default=5.0, help="Axis probe offset in degrees")
    parser.add_argument("--servo-test-wait", type=float, default=0.6, help="Seconds between axis probe commands")
    parser.add_argument("--sync-capture", action="store_true", help="Disable latest-frame camera thread for comparison")
    parser.add_argument("--capture-buffer", type=int, default=1, help="Requested live-camera driver buffer; backend may reject it")
    parser.add_argument("--frame-timeout", type=float, default=2.0, help="Maximum wait for a fresh live-camera frame")
    parser.add_argument("--servo-range", type=float, default=270.0, help="Physical travel for the pulse endpoints in degrees; PM155 default: 270")
    parser.add_argument("--pulse-min", type=int, default=500, help="Calibrated pulse minimum in microseconds; PM155 default: 500")
    parser.add_argument("--pulse-max", type=int, default=2500, help="Calibrated pulse maximum in microseconds; PM155 default: 2500")
    parser.add_argument("--no-display", action="store_true")
    stats_options = parser.add_mutually_exclusive_group()
    stats_options.add_argument("--hide-stats", action="store_true", default=True, help="Hide the statistics panel (default); keep boxes and the image-center crosshair")
    stats_options.add_argument("--show-stats", dest="hide_stats", action="store_false", help="Show the statistics panel")
    parser.add_argument("--display-max-width", type=int, default=960, help="Maximum view width; inference/control use the original image")
    parser.add_argument("--display-max-height", type=int, default=540, help="Maximum view height; display scaling preserves aspect ratio")
    video_options = parser.add_mutually_exclusive_group()
    video_options.add_argument("--no-video", action="store_true", default=True, help="Do not save video (default)")
    video_options.add_argument("--save-video", dest="no_video", action="store_false", help="Save the annotated video")
    parser.add_argument("--video-fps", type=float, default=None, help="Saved video playback rate; NOT measured FPS")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no frame limit")
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser


def servo_config_from_args(args):
    return ServoConfig(
        pan_channel=args.pan_channel, tilt_channel=args.tilt_channel,
        pan_home=args.pan_home, tilt_home=args.tilt_home,
        pan_min=args.pan_min, pan_max=args.pan_max,
        tilt_min=args.tilt_min, tilt_max=args.tilt_max,
        pan_direction=args.pan_direction, tilt_direction=args.tilt_direction,
        pan_gain=args.pan_gain, tilt_gain=args.tilt_gain,
        max_speed=args.servo_max_speed, max_step=args.servo_max_step,
        deadzone=args.deadzone, smoothing=args.servo_smoothing,
        target_hold=args.target_hold, target_gate=args.target_gate,
        pulse_min=args.pulse_min, pulse_max=args.pulse_max, actuation_range=args.servo_range,
        pan_ki=args.pan_ki, tilt_ki=args.tilt_ki, pan_kd=args.pan_kd, tilt_kd=args.tilt_kd,
        pid_integral_limit=args.pid_integral_limit,
        pid_derivative_smoothing=args.pid_derivative_smoothing, pid_reset_gap=args.pid_reset_gap,
        velocity_feedforward=args.velocity_feedforward, horizontal_fov=args.horizontal_fov,
        velocity_smoothing=args.velocity_smoothing, feedforward_max_speed=args.feedforward_max_speed,
        feedforward_deadband=args.feedforward_deadband,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_identity_memory:
        args.single_target_recovery = False
    if not 0 <= args.deadzone < 1 or args.settle_frames < 1 or args.lost_frames < 1:
        parser.error("Require 0 <= deadzone < 1, settle-frames >= 1, lost-frames >= 1")
    if args.fps_window < 2 or not 0 <= args.confThreshold <= 1 or args.max_frames < 0:
        parser.error("Invalid FPS window, confidence threshold or frame limit")
    if args.video_fps is not None and (not math.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error("video-fps must be a finite positive number")
    if not math.isfinite(args.home_wait) or args.home_wait < 0:
        parser.error("home-wait must be finite and nonnegative")
    if (args.home_only or args.servo_test) and args.no_servo:
        parser.error("home-only/servo-test cannot be used with no-servo")
    if args.home_only and args.servo_test:
        parser.error("Choose either home-only or servo-test")
    if not math.isfinite(args.servo_test_step) or not 0 < args.servo_test_step <= 10:
        parser.error("servo-test-step must be > 0 and <= 10 degrees")
    if not math.isfinite(args.servo_test_wait) or args.servo_test_wait < 0:
        parser.error("servo-test-wait must be finite and nonnegative")
    if args.capture_buffer < 1 or not math.isfinite(args.frame_timeout) or args.frame_timeout <= 0:
        parser.error("Invalid capture buffer or frame timeout")
    if args.display_max_width < 1 or args.display_max_height < 1:
        parser.error("Display dimensions must be positive")
    if (not all(math.isfinite(value) for value in (args.detect_interval, args.lock_prediction, args.lock_release))
            or args.detect_interval <= 0 or not 0 <= args.lock_prediction < args.lock_release):
        parser.error("Require detect-interval > 0 and 0 <= lock-prediction < lock-release")
    if args.legacy_single_target and not args.legacy_fusion and args.lock_prediction > .35:
        parser.error('Source-aware fusion requires --lock-prediction <= 0.35 seconds')
    if (args.detector_threads < 0 or args.opencv_threads < 0 or args.recovery_confirmations < 2
            or not all(math.isfinite(v) for v in (args.max_detection_age, args.recovery_timeout, args.appearance_threshold))
            or args.max_detection_age <= 0 or args.recovery_timeout < args.lock_release
            or not 0 < args.appearance_threshold <= 1):
        parser.error("Invalid asynchronous detection or identity recovery configuration")
    if (not all(math.isfinite(v) for v in (args.recovery_gate,args.recovery_confirm_window,args.recovery_area_ratio))
            or not args.target_gate <= args.recovery_gate <= max(args.target_gate,1.)
            or args.recovery_confirm_window <= 0 or not 1 <= args.recovery_area_ratio <= 4):
        parser.error('Invalid recovery gate, confirmation window or area ratio')
    if (not all(math.isfinite(v) for v in (args.detector_reanchor_after,args.detector_reanchor_confidence))
            or args.detector_reanchor_after <= 0 or not 0 <= args.detector_reanchor_confidence <= 1):
        parser.error('Invalid detector reanchor age/confidence')
    mot_values = (args.mot_history_seconds, args.mot_search_interval, args.mot_birth_confidence,
                  args.mot_low_score_threshold,
                  args.mot_birth_center_separation_fraction,
                  args.mot_birth_center_max_sigma_fraction,
                  args.mot_tombstone_retention_seconds,
                  args.mot_lost_after, args.mot_tentative_timeout, args.mot_retain_seconds,
                  args.mot_prediction_horizon, args.mot_detector_anchor_horizon, args.mot_active_hold_seconds,
                  args.mot_switch_confidence, args.mot_switch_window,
                  args.mot_mahalanobis_gate, args.mot_replay_budget_ms,
                  args.mot_jump_center_ratio, args.mot_jump_speed_short_sides,
                  args.mot_semantic_soft_age, args.mot_semantic_hard_age,
                  args.lk_centroid_soft_limit, args.lk_center_soft_limit,
                  args.lk_velocity_soft_limit, args.lk_velocity_sigma)
    if all(math.isfinite(value) for value in mot_values):
        if args.mot_tombstone_retention_seconds < args.mot_history_seconds:
            parser.error('--mot-tombstone-retention-seconds must be at least --mot-history-seconds')
        if args.mot_lost_after >= args.mot_prediction_horizon:
            parser.error('--mot-lost-after must be below --mot-prediction-horizon so lost tracks have a bounded prediction window')
    if (not all(math.isfinite(value) for value in mot_values)
            or args.mot_max_tracks < 1 or args.mot_max_detections < 1
            or args.mot_confirm_hits < 2 or args.mot_history_seconds < .5
            or args.mot_search_interval <= 0 or args.mot_tentative_timeout <= 0
            or not 0 <= args.mot_birth_confidence <= 1 or args.mot_lost_after <= 0
            or not 0 <= args.mot_low_score_threshold < 1
            or not 0 < args.mot_birth_center_separation_fraction <= 1
            or not 0 < args.mot_birth_center_max_sigma_fraction <= 1
            or args.mot_retain_seconds < args.mot_active_hold_seconds
            or not 0 < args.mot_prediction_horizon <= .35
            or args.mot_detector_anchor_horizon < args.mot_prediction_horizon
            or args.mot_history_seconds < args.mot_prediction_horizon
            or args.mot_active_hold_seconds < args.mot_prediction_horizon
            or not 0 <= args.mot_switch_confidence <= 1
            or args.mot_switch_confirmations < 2 or args.mot_switch_window <= 0
            or args.mot_mahalanobis_gate <= 0 or args.mot_replay_budget_ms <= 0
            or args.mot_jump_center_ratio <= 0 or args.mot_jump_speed_short_sides <= 0
            or args.mot_semantic_soft_age <= 0
            or args.mot_semantic_hard_age <= args.mot_semantic_soft_age
            or args.mot_semantic_misses < 1
            or min(args.lk_centroid_soft_limit, args.lk_center_soft_limit,
                   args.lk_velocity_soft_limit, args.lk_velocity_sigma) <= 0):
        parser.error('Invalid multi-target or LK fusion configuration')
    if (not args.legacy_single_target
            and args.mot_low_score_threshold >= args.confThreshold):
        parser.error('--mot-low-score-threshold must be below --confThreshold in MOT mode')
    if args.mot_click_select and args.no_display:
        parser.error('--mot-click-select requires the display window')
    try:
        from roi_affine_tracker import FlowConfig
        flow_config = FlowConfig(center_only=not args.legacy_fusion,
                                 roi_scale=args.flow_roi_scale, min_inliers=args.flow_min_inliers,
                                 min_inlier_ratio=args.flow_min_inlier_ratio)
        if (not 1 <= args.alignment_max_steps <= 8 or not math.isfinite(args.alignment_budget_ms)
                or args.alignment_budget_ms <= 0):
            raise ValueError('Require alignment-max-steps in 1..8 and a positive finite alignment-budget-ms')
        servo_config = servo_config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.home_only and not args.servo_test:
        for path in (args.modelpath, args.classfile):
            if not path.is_file():
                parser.error("File not found: %s" % path)

    log_dir = args.log_dir or Path(__file__).resolve().parent / "logs" / "dj_show2" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir.mkdir(parents=True, exist_ok=True)
    detector_debug_dir = log_dir / "detector_debug"
    if args.mot_debug_detections:
        detector_debug_dir.mkdir(parents=True, exist_ok=True)
    detector_debug_details = dict(enabled=bool(args.mot_debug_detections),
                                  directory=str(detector_debug_dir) if args.mot_debug_detections else None,
                                  source_frames_saved=0, save_errors=0)
    click_state = {"tracks": (), "scale_x": 1.0, "scale_y": 1.0,
                   "left_button_event": None, "request_track_id": None,
                   "click_feedback": "click a confirmed track"}
    mouse_callback_registered = False
    scene_metrics = SceneMetrics(args.deadzone, args.settle_frames, args.lost_frames)
    frame_metrics = None
    capture = writer = cv2 = None
    frame_reader = None
    capture_details = {"mode": "not_opened"}
    image_geometry = display_geometry = None
    servo_test_commands = []
    control = None
    control_state = {"control_status": "disabled"}
    visual_lock = None
    visual_backend = "disabled"
    mot_tracker = target_selector = active_frontend = None
    last_selection = None
    last_mot_detection_requested_at = None
    mot_mode = not args.legacy_single_target
    stream_epoch = 0
    mot_details = {"mode": "legacy_single_target" if not mot_mode else "initializing"}
    detection_worker = None
    detection_details = {"mode": "synchronous"}
    projector = None
    model_details = {}
    reason, failure = "completed", None
    source = int(args.source) if args.source.isdecimal() else args.source
    print("dj_show2 version: %s" % VERSION)
    print("Logs: %s" % log_dir.resolve())
    print("Tracking mode: %s" % ("multi_target_single_active" if mot_mode else "3.8_legacy_single_target"))
    try:
        if not args.no_servo:
            control = open_servo_controller(servo_config)
            print("Servo mapping: range=%.1f deg, pulse=%s..%s us" % (
                servo_config.actuation_range, servo_config.pulse_min, servo_config.pulse_max), flush=True)
            print("PID Kp/Ki/Kd: pan=%.2f/%.2f/%.2f tilt=%.2f/%.2f/%.2f; max_speed=%.1f max_step=%.1f" % (
                servo_config.pan_gain, servo_config.pan_ki, servo_config.pan_kd,
                servo_config.tilt_gain, servo_config.tilt_ki, servo_config.tilt_kd,
                servo_config.max_speed, servo_config.max_step), flush=True)
            if servo_config.velocity_feedforward:
                print("Velocity feedforward: gain=%.2f HFOV=%.1f cap=%.1f deg/s; command-pose proxy, NOT measured target speed" % (
                    servo_config.velocity_feedforward, servo_config.horizontal_fov,
                    servo_config.feedforward_max_speed), flush=True)
            control_state = control.initialize(time.perf_counter())
            print("HOME COMMANDED: pan[channel=%d]=%.1f deg, tilt[channel=%d]=%.1f deg; directions=%+d/%+d" % (
                args.pan_channel, args.pan_home, args.tilt_channel, args.tilt_home,
                args.pan_direction, args.tilt_direction), flush=True)
            time.sleep(args.home_wait)
        if args.home_only:
            reason = "home_only"
            return 0
        if args.servo_test:
            home = args.pan_home if args.servo_test == "pan" else args.tilt_home
            channel = args.pan_channel if args.servo_test == "pan" else args.tilt_channel
            for requested in (home + args.servo_test_step, home, home - args.servo_test_step, home):
                control_state = control.command_axis(args.servo_test, requested)
                actual = control_state[args.servo_test + "_angle_deg"]
                servo_test_commands.append({"axis": args.servo_test, "channel": channel,
                                            "requested_deg": requested, "commanded_deg": actual})
                print("AXIS TEST: %s channel=%d requested=%.2f commanded=%.2f deg" % (
                    args.servo_test, channel, requested, actual), flush=True)
                time.sleep(args.servo_test_wait)
            reason = "servo_test"
            return 0
        detector_output_floor = (args.mot_low_score_threshold if mot_mode
                                 else args.confThreshold)
        detector = yolov5_lite(args.modelpath, args.classfile, detector_output_floor, args.detector_threads,
                               assign_ids=(args.legacy_single_target and args.no_visual_lock),
                               ort_spinning=args.ort_spinning)
        cv2 = detector.cv2
        if callable(getattr(cv2, 'setNumThreads', None)):
            cv2.setNumThreads(args.opencv_threads)
        with args.modelpath.open('rb') as model_stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: model_stream.read(1024 * 1024), b''):
                digest.update(chunk)
        model_details = dict(sha256=digest.hexdigest(), bytes=args.modelpath.stat().st_size,
                             input_width=getattr(detector, 'input_w', None),
                             input_height=getattr(detector, 'input_h', None),
                             detector_output_floor=detector_output_floor,
                             mot_high_score_threshold=(args.confThreshold if mot_mode else None),
                             ort_intra_op_threads=args.detector_threads,
                             ort_spinning_requested=args.ort_spinning,
                             ort_spinning_applied=getattr(detector, 'ort_spinning_applied', False),
                             opencv_threads=args.opencv_threads)
        factory = None
        if mot_mode:
            from multi_target_tracker import MultiTargetTracker
            from target_selector import TargetSelector
            mot_tracker = MultiTargetTracker(
                max_tracks=args.mot_max_tracks,
                max_assignment_detections=args.mot_max_detections,
                history_seconds=args.mot_history_seconds,
                confirm_hits=args.mot_confirm_hits,
                birth_confidence=args.mot_birth_confidence,
                birth_center_separation_fraction=args.mot_birth_center_separation_fraction,
                birth_center_max_sigma_fraction=args.mot_birth_center_max_sigma_fraction,
                tombstone_retention_seconds=args.mot_tombstone_retention_seconds,
                lost_after_seconds=args.mot_lost_after,
                retain_seconds=args.mot_retain_seconds,
                tentative_timeout_seconds=args.mot_tentative_timeout,
                prediction_horizon=args.mot_prediction_horizon,
                detector_anchor_horizon=args.mot_detector_anchor_horizon,
                mahalanobis_gate=args.mot_mahalanobis_gate,
                replay_budget_ms=args.mot_replay_budget_ms,
                max_result_age=min(args.max_detection_age, args.mot_prediction_horizon),
                rd_calibration_path=args.mot_rd_calibration,
                debug_association_matrix=args.mot_debug_association_matrix,
                epoch=stream_epoch,
                high_score_threshold=args.confThreshold,
                low_score_threshold=args.mot_low_score_threshold,
                hard_jump_center_ratio=args.mot_jump_center_ratio,
                hard_jump_speed_short_sides=args.mot_jump_speed_short_sides,
                semantic_soft_age=args.mot_semantic_soft_age,
                semantic_hard_age=args.mot_semantic_hard_age,
                semantic_miss_count=args.mot_semantic_misses)
            target_selector = TargetSelector(
                control_horizon=args.mot_prediction_horizon,
                active_hold_seconds=args.mot_active_hold_seconds,
                switch_confidence=args.mot_switch_confidence,
                switch_confirmations=args.mot_switch_confirmations,
                switch_window_seconds=args.mot_switch_window)
            if not args.no_visual_lock:
                from visual_target_lock import create_tracker_factory
                from active_track_frontend import ActiveTrackFrontend
                factory, visual_backend = create_tracker_factory(
                    cv2, args.visual_tracker, flow_config=flow_config)
                template_bank = None
                if not args.no_template_recovery and callable(getattr(cv2, "matchTemplate", None)):
                    from protected_template import ProtectedTemplateBank
                    template_bank = ProtectedTemplateBank(cv2)
                active_frontend = ActiveTrackFrontend(
                    factory, template_bank=template_bank,
                    offset_soft_limit=args.lk_centroid_soft_limit,
                    center_soft_limit=args.lk_center_soft_limit,
                    velocity_soft_limit=args.lk_velocity_soft_limit,
                    base_velocity_sigma_px_s=args.lk_velocity_sigma)
            else:
                factory = None
                visual_backend = "disabled"
            mot_details["mode"] = "multi_target_single_active"
        elif not args.no_visual_lock:
            from visual_target_lock import TargetLock, create_tracker_factory
            if not args.legacy_fusion:
                from fusion_target_lock import FusionTargetLock
                TargetLock = FusionTargetLock
            factory, visual_backend = create_tracker_factory(cv2, args.visual_tracker, flow_config=flow_config)
            if factory is not None:
                appearance = None
                fusion_options={}
                if not args.legacy_fusion:
                    template_bank=None
                    if not args.no_template_recovery:
                        if all(callable(getattr(cv2,name,None)) for name in ('resize','matchTemplate')):
                            from protected_template import ProtectedTemplateBank
                            template_bank=ProtectedTemplateBank(cv2)
                        else:
                            print('Template recovery unavailable: OpenCV template matching is missing',flush=True)
                    fusion_options['template_bank']=template_bank
                if all(callable(getattr(cv2, name, None)) for name in ('cvtColor', 'resize', 'calcHist')):
                    from target_appearance import PatchAppearance, create_box_projector
                    if args.legacy_fusion:
                        projector = create_box_projector(cv2, flow_config, history_seconds=args.max_detection_age+.1,
                                                     max_steps=args.alignment_max_steps,
                                                     budget_ms=args.alignment_budget_ms,
                                                     legacy=visual_backend == 'flow-legacy')
                    if not args.no_identity_memory and not args.single_target_recovery:
                        appearance = PatchAppearance(cv2)
                visual_lock = TargetLock(factory, args.detect_interval, args.lock_prediction,
                                         args.lock_release, args.target_gate, args.pid_reset_gap,
                                         appearance=appearance, projector=projector,
                                         recovery_timeout=args.recovery_timeout,
                                         appearance_threshold=args.appearance_threshold,
                                         recovery_confirmations=args.recovery_confirmations,
                                         max_detection_age=args.max_detection_age,
                                         single_target_recovery=args.single_target_recovery,
                                         recovery_gate=args.recovery_gate,
                                         recovery_confirm_window=args.recovery_confirm_window,
                                         recovery_area_ratio=args.recovery_area_ratio,
                                         detector_reanchor=not args.no_detector_reanchor,
                                         detector_reanchor_after=args.detector_reanchor_after,
                                         detector_reanchor_confidence=args.detector_reanchor_confidence,
                                         use_sort=True,**fusion_options)
            else:
                print("Visual tracking unavailable in this OpenCV build; using detection-only positions", flush=True)
                if callable(getattr(detector, 'enable_sort', None)):
                    detector.enable_sort()
        print("Visual/LK backend: %s; method=%s; YOLO interval=%.3fs" % (
            visual_backend, getattr(factory, 'method', visual_backend) if factory is not None else 'disabled',
            args.detect_interval), flush=True)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError("Cannot open video source: %s" % args.source)
        output_fps = args.video_fps if args.video_fps is not None else capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(output_fps) or output_fps <= 0:
            output_fps = 20.0
        live_source = isinstance(source, int) or str(source).startswith("/dev/video")
        capture_details = {"mode": "synchronous", "buffer_set_accepted": None}
        if live_source:
            buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
            if buffer_property is not None:
                try:
                    capture_details["buffer_set_accepted"] = bool(capture.set(buffer_property, args.capture_buffer))
                except Exception as exc:
                    capture_details["buffer_set_accepted"] = False
                    capture_details["buffer_set_error"] = str(exc)
            if not args.sync_capture:
                reader = LatestFrameCapture(capture).start()
                frame_reader = reader
                capture_details["mode"] = "latest_frame"
        print("Capture mode: %s; buffer request accepted: %s" % (
            capture_details["mode"], capture_details["buffer_set_accepted"]), flush=True)

        def detect_targets(image, context=None):
            roi = (context or {}).get("search_roi")
            if roi is not None:
                left, top, right, bottom = roi
                detected, identified, elapsed = detector.detect(image[top:bottom, left:right])
                detected = remap_roi_detections(detected, roi)
            else:
                detected, identified, elapsed = detector.detect(image)
            return ([item for item in detected if args.target_class is None or item.class_id == args.target_class],
                    identified, elapsed)

        search_zoom_next = False

        def submit_detection_frame(frame, packet):
            nonlocal search_zoom_next
            request_age_ms = (time.perf_counter()-packet.received_at)*1000
            submit_reason = detector_submission_reason(
                detection_worker.stats(), request_age_ms/1000,
                min(.1,args.max_detection_age/2), args.detector_prefetch)
            if submit_reason != 'submitted':
                return False, submit_reason, request_age_ms, 0.
            submitted_at = time.perf_counter()
            if mot_mode:
                context = dict(stream_epoch=stream_epoch,
                               active_epoch=target_selector.active_epoch,
                               active_track_id=target_selector.active_track_id,
                               width=int(frame.shape[1]), height=int(frame.shape[0]))
                if mot_search_needed(last_selection):
                    search_zoom_next = not search_zoom_next
                    if search_zoom_next:
                        possible = [view for view in mot_tracker.snapshot(packet.received_at)
                                    if mot_search_roi_eligible(view)]
                        if possible:
                            focused = max(possible, key=lambda view: (
                                view.track_id == target_selector.active_track_id,
                                view.last_visual_support_at, view.confidence))
                            x1, y1, x2, y2 = focused.bbox
                            cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
                            side = max(160.0, 3.0 * max(x2 - x1, y2 - y1))
                            side = min(side, frame.shape[0], frame.shape[1])
                            left = int(max(0, min(frame.shape[1] - side, cx - side * .5)))
                            top = int(max(0, min(frame.shape[0] - side, cy - side * .5)))
                            context["search_roi"] = (left, top, int(left + side), int(top + side))
            else:
                context = visual_lock.detection_context(packet.received_at)
            detection_worker.submit(frame, packet.sequence, packet.received_at,
                                    context)
            if visual_lock is not None:
                visual_lock.mark_detection_requested(packet.received_at)
            return True, submit_reason, request_age_ms, (time.perf_counter()-submitted_at)*1000

        if ((mot_mode or (visual_lock is not None and (not args.legacy_fusion or projector is not None)))
                and not args.sync_detect and (live_source or args.async_detect)):
            detection_worker = LatestDetectionWorker(
                detect_targets, with_context=mot_mode).start()
            detection_details["mode"] = "asynchronous"
        elif args.async_detect and not mot_mode:
            raise RuntimeError('Asynchronous detection requires an available visual tracker and OpenCV image alignment')
        print("Detection mode: %s; identity policy=%s; max result age=%.3fs" % (
            detection_details["mode"], "multi_target_kf" if mot_mode else (
                visual_lock.summary()['identity_policy'] if visual_lock else 'detection_only'),
            min(args.max_detection_age, args.mot_prediction_horizon) if mot_mode else args.max_detection_age), flush=True)
        print("Model input: %sx%s; ORT threads=%d (0=auto); OpenCV threads=%d; SHA256=%s" % (
            model_details['input_width'], model_details['input_height'], args.detector_threads,
            args.opencv_threads, model_details['sha256'][:16]), flush=True)
        print('Model file: %s; confidence=%.3f; target_class=%s' % (
            args.modelpath.resolve(), args.confThreshold, args.target_class), flush=True)
        if mot_mode:
            print("MOT score stages: high>=%.3f; low>=%.3f (low detections update established tracks only); birth>=%.3f" % (
                args.confThreshold, args.mot_low_score_threshold,
                args.mot_birth_confidence), flush=True)
            if args.mot_debug_detections:
                print("Source-frame detector diagnostics: %s" % detector_debug_dir.resolve(), flush=True)
        frame_metrics = FrameMetrics(time.perf_counter(), args.fps_window)
        last_print = frame_metrics.started_at
        previous_sequence = None
        last_semantic_key = None
        with CsvLog(log_dir / "frames.csv") as csv_log:
            while True:
                started = time.perf_counter()
                if frame_reader is not None:
                    ok, packet = frame_reader.read(timeout=args.frame_timeout)
                    frame = packet.frame if ok else None
                else:
                    ok, frame = capture.read()
                    packet = FramePacket(frame, frame_metrics.frame_count + 1, started, time.perf_counter())
                if not ok:
                    reason = "capture_read_failed" if isinstance(source, int) else "end_of_video"
                    break
                seq_delta = None if previous_sequence is None else packet.sequence - previous_sequence
                previous_sequence = packet.sequence
                video_frame_index = frame_metrics.frame_count + 1 if not args.no_video else None
                captured_at = time.perf_counter()
                detector_poll_ms = detector_submit_ms = 0.
                height, width = frame.shape[:2]
                geometry = {"width": width, "height": height, "center_x": width // 2, "center_y": height // 2}
                if geometry != image_geometry:
                    if mot_mode and image_geometry is not None:
                        stream_epoch += 1
                        mot_tracker.reset(epoch=stream_epoch)
                        target_selector.reset()
                        last_selection = None
                        last_semantic_key = None
                        last_mot_detection_requested_at = None
                        if active_frontend is not None:
                            active_frontend.reset(reset_bank=True)
                    image_geometry = geometry
                    print("Source image: width=%d height=%d center=(%d,%d)" % (
                        width, height, width // 2, height // 2), flush=True)
                if mot_tracker is not None:
                    frame_at = float(packet.received_at)
                    previous_views = mot_tracker.snapshot() if mot_tracker._latest_timestamp is not None else []
                    previous_by_id = {view.track_id: view for view in previous_views}
                    active_before = target_selector.active_track_id
                    lk_event = None
                    lk_diag = active_frontend.last_diagnostics if active_frontend is not None else {}
                    if active_frontend is not None and active_before in previous_by_id:
                        lk_event, lk_diag = active_frontend.observe_interval(
                            frame, frame_at, packet.sequence, previous_by_id[active_before],
                            target_selector.active_epoch)
                    mot_tracker.advance(frame_at, packet.sequence,
                                        lk_observations=(lk_event,) if lk_event is not None else (),
                                        frame_shape=frame.shape[:2])

                    result = None
                    submitted = False
                    submit_reason = "synchronous" if detection_worker is None else "not_due"
                    request_age_ms = None
                    inference_ms = None
                    result_age_ms = None
                    raw_detections = []
                    source_frame = frame
                    source_timestamp = frame_at
                    source_sequence = packet.sequence
                    source_active_epoch = target_selector.active_epoch
                    detector_ran = False
                    detector_accepted = False
                    association_result = dict(accepted=False, reason="not_run", diagnostics={})
                    if detection_worker is not None:
                        poll_started = time.perf_counter()
                        result = detection_worker.poll()
                        detector_poll_ms = (time.perf_counter() - poll_started) * 1000.0
                        if result is not None:
                            detector_ran = True
                            raw_detections = result.detections
                            source_frame = result.frame
                            source_timestamp = float(result.received_at)
                            source_sequence = int(result.sequence)
                            source_active_epoch = int(result.context.get("active_epoch", 0))
                            inference_ms = result.inference_ms
                            result_age_ms = max(0.0, (frame_at - source_timestamp) * 1000.0)
                            association_result = mot_tracker.update_detections(
                                raw_detections, source_timestamp, source_sequence,
                                epoch=result.context.get("stream_epoch", -1),
                                frame_shape=result.frame.shape[:2],
                                active_epoch=source_active_epoch,
                                full_frame=result.context.get("search_roi") is None)
                            detector_accepted = bool(association_result.get("accepted"))
                        search_interval = min(args.detect_interval, args.mot_search_interval)\
                            if last_selection is None or last_selection.target is None else args.detect_interval
                        due = (last_mot_detection_requested_at is None
                               or frame_at - last_mot_detection_requested_at >= search_interval)
                        if due:
                            submitted, submit_reason, request_age_ms, detector_submit_ms = submit_detection_frame(frame, packet)
                            if submitted:
                                last_mot_detection_requested_at = frame_at
                    else:
                        search_interval = min(args.detect_interval, args.mot_search_interval)\
                            if last_selection is None or last_selection.target is None else args.detect_interval
                        due = (last_mot_detection_requested_at is None
                               or frame_at - last_mot_detection_requested_at >= search_interval)
                        if due:
                            detector_ran = True
                            raw_detections, _, inference_ms = detect_targets(frame)
                            association_result = mot_tracker.update_detections(
                                raw_detections, frame_at, packet.sequence,
                                epoch=stream_epoch, frame_shape=frame.shape[:2],
                                active_epoch=target_selector.active_epoch)
                            detector_accepted = bool(association_result.get("accepted"))
                            last_mot_detection_requested_at = frame_at

                    accepted_diag = association_result.get("diagnostics", {}) if detector_accepted else {}
                    capacity_rejected_candidates = sum(
                        1 for reason in accepted_diag.get("association_rejections", {}).values()
                        if reason == "track_capacity") + int(
                            accepted_diag.get("replayed_capacity_rejections_added", 0))
                    capacity_log = mot_tracker.capacity_log_state(
                        frame_at, rejected_candidates=capacity_rejected_candidates)
                    if capacity_log["event"]:
                        print("MOT track capacity %s: tracks=%d/%d, rejected_since_summary=%d" % (
                            capacity_log["event"], capacity_log["track_count"],
                            capacity_log["max_tracks"], capacity_log["rejected_candidates"]),
                            flush=True)

                    views = mot_tracker.snapshot(frame_at)
                    view_by_id = {view.track_id: view for view in views}
                                                                           
                                                                              
                    control_at = time.perf_counter()
                    control_views = mot_tracker.snapshot(control_at)
                    click_state["tracks"] = tuple(views)
                    requested_track_id = click_state.get("request_track_id")
                    if args.mot_click_select and requested_track_id is not None:
                        click_state["request_track_id"] = None
                        target_selector.request_acquire(requested_track_id)
                    active_track_retained = mot_tracker.contains_track(
                        target_selector.active_track_id)
                    selection = target_selector.select(
                        control_views, control_at,
                        active_track_retained=active_track_retained)
                    if selection.manual_request_result:
                        click_state["click_feedback"] = "%s:%s" % (
                            selection.manual_request_result,
                            selection.active_track_id or "none")
                    if selection.switched and control is not None:
                        control.reset_target_lock()
                    mot_tracker.set_active_track_id(selection.active_track_id)
                    active_view = view_by_id.get(selection.active_track_id)
                    active_control_view = next(
                        (view for view in control_views
                         if view.track_id == selection.active_track_id), None)
                    semantic_key = (selection.active_track_id,
                                    active_control_view.semantic_state
                                    if active_control_view is not None else "none")
                    semantic_transition = (
                        "" if semantic_key == last_semantic_key else
                        "%s:%s->%s:%s" % (
                            *(last_semantic_key or (None, "none")), *semantic_key))
                    last_semantic_key = semantic_key
                    if active_frontend is not None:
                        if (active_control_view is not None
                                and active_control_view.semantic_state == "invalid"):
                            if active_frontend.track_id is not None:
                                active_frontend.reset(reset_bank=True)
                        elif selection.active_track_id != active_frontend.track_id:
                            if active_view is not None:
                                active_frontend.bind(frame, active_view, frame_at, packet.sequence,
                                                     selection.active_epoch)
                            else:
                                active_frontend.reset(reset_bank=True)
                        elif active_view is not None and active_frontend.tracker is None:
                            active_frontend.bind(frame, active_view, frame_at, packet.sequence,
                                                 selection.active_epoch)
                    if detector_accepted and active_frontend is not None:
                        assignments = association_result.get("diagnostics", {}).get("association_assignments", {})
                        for index, identity in assignments.items():
                            if str(identity) != str(selection.active_track_id):
                                continue
                            try:
                                item = raw_detections[int(index)]
                            except (IndexError, TypeError, ValueError):
                                continue
                            active_view = view_by_id.get(str(identity))
                            if active_view is not None:
                                seed_diag = active_frontend.record_detector(
                                    source_frame, active_view, item.bbox, item.confidence,
                                    source_timestamp, source_sequence,
                                    selection.active_epoch,
                                    current_frame=source_sequence == packet.sequence)
                                lk_diag = dict(lk_diag or {})
                                lk_diag.update(template_seed_reject_reason=seed_diag.get("seed_reject_reason"),
                                               template_anchor_score=seed_diag.get("anchor_score"),
                                               template_best_score=seed_diag.get("best_score"))
                    last_selection = selection
                    candidates = [selection.target] if selection.target is not None else []
                    measured = [target for target in candidates
                                if target.position_source == "detector"]
                                                                             
                                                                      
                    control_started = time.perf_counter()
                    if control is not None:
                        control_state = control.update(
                            candidates, width, height, control_at,
                            observed_at=(candidates[0].last_visual_support_at
                                         if candidates else packet.received_at),
                            hold_reason=("ambiguous" if selection.status == "ambiguous" else
                                         "position_invalid" if selection.manual_active_lock
                                         and selection.target is None else None),
                            selected_track_id=selection.active_track_id,
                            preserve_target_identity=selection.manual_active_lock)
                    control_done = time.perf_counter()
                    detections = views
                                                                           
                                                                         
                                                                           
                    replay_diag = (association_result.get("diagnostics", {})
                                   if detector_ran else {})
                    assigned_ids = replay_diag.get("association_assignments", {}) if detector_accepted else {}
                    association_summaries = (replay_diag.get("association_detection_summaries", {})
                                             if detector_accepted else {})
                    association_diag = replay_diag if detector_accepted else {}
                    report_capacity_detail = bool(capacity_log["event"])
                    if not report_capacity_detail:
                        association_summaries = {
                            key: ({**value, "rejection_reason": None}
                                  if value.get("rejection_reason") == "track_capacity" else value)
                            for key, value in association_summaries.items()}
                    association_rejections_for_log = dict(
                        association_diag.get("association_rejections", {}))
                    if not report_capacity_detail:
                        association_rejections_for_log = {
                            key: value for key, value in association_rejections_for_log.items()
                            if value != "track_capacity"}
                    detection_track_rows = []
                    for index, item in enumerate(raw_detections):
                        identity = assigned_ids.get(index, assigned_ids.get(str(index)))
                        assigned_view = view_by_id.get(str(identity)) if identity is not None else None
                        association = association_summaries.get(str(index), {})
                        detection_track_rows.append(dict(
                            source_index=index, bbox=list(item.bbox),
                            confidence=float(item.confidence), class_id=int(item.class_id),
                            track_id=str(identity) if identity is not None else None,
                            track_status=assigned_view.status if assigned_view is not None else "unassigned",
                            best_track_id=association.get("best_track_id"),
                            best_cost=association.get("best_cost"),
                            second_track_id=association.get("second_track_id"),
                            second_cost=association.get("second_cost"),
                            best_second_margin=association.get("best_second_margin"),
                            score_stage=association.get("score_stage", "not_associated"),
                            association_stage=association.get("association_stage", "unmatched"),
                            association_success=association.get("association_success", False),
                            detector_velocity_policy=association.get("detector_velocity_policy"),
                            detector_velocity_guard_reason=association.get("detector_velocity_guard_reason"),
                            detector_center_innovation_ratio=association.get("detector_center_innovation_ratio"),
                            detector_anchor_speed_short_sides_s=association.get("detector_anchor_speed_short_sides_s"),
                            track_created=association.get("track_created", False),
                            duplicate_parent_source_index=association.get("duplicate_parent_source_index"),
                            birth_guard_track_id=association.get("birth_guard_track_id"),
                            birth_guard_center_distance_px=association.get(
                                "birth_guard_center_distance_px"),
                            birth_guard_position_sigma_px=association.get(
                                "birth_guard_position_sigma_px"),
                            birth_guard_position_sigma_x_px=association.get(
                                "birth_guard_position_sigma_x_px"),
                            birth_guard_position_sigma_y_px=association.get(
                                "birth_guard_position_sigma_y_px"),
                            birth_guard_evaluated=association.get(
                                "birth_guard_evaluated", False),
                            birth_guard_skip_reason=association.get(
                                "birth_guard_skip_reason", ""),
                            birth_guard_skipped_uncertain=association.get(
                                "birth_guard_skipped_uncertain", False),
                            association_rejection_reason=association.get("rejection_reason")))
                    if args.mot_debug_detections and detector_ran:
                        debug_frame = source_frame.copy()
                        cv2.putText(debug_frame,
                                    "YOLO source=%d age=%.1fms candidates=%d" % (
                                        source_sequence, result_age_ms or 0.0, len(detection_track_rows)),
                                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        if not detection_track_rows:
                            cv2.putText(debug_frame, "NO DETECTIONS ABOVE LOW SCORE FLOOR",
                                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                        (0, 0, 255), 1, cv2.LINE_AA)
                        for row in detection_track_rows:
                            x1, y1, x2, y2 = map(int, row["bbox"])
                            stage = row.get("association_stage", "unmatched")
                            color = (0, 220, 0) if row.get("track_id") else (0, 165, 255)
                            label = "c%d %.2f %s %s" % (
                                row["class_id"], row["confidence"],
                                row.get("track_id") or "unassigned", stage)
                            reject = row.get("association_rejection_reason")
                            if reject and reject not in ("new_track", "matched_existing"):
                                label += " " + str(reject)
                            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(debug_frame, label, (x1, max(14, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
                        debug_path = detector_debug_dir / ("source_%08d.jpg" % source_sequence)
                        if cv2.imwrite(str(debug_path), debug_frame):
                            detector_debug_details["source_frames_saved"] += 1
                        else:
                            detector_debug_details["save_errors"] += 1
                    matched_detection_by_track = {
                        str(identity): int(index) for index, identity in assigned_ids.items()}
                    lk_diag = dict(lk_diag or {})
                    active_position_reason = next((item["reason"] for item in selection.candidate_diagnostics
                                                   if item["track_id"] == selection.active_track_id), None)
                    lock_data = dict(
                        detector_ran=detector_ran,
                        raw_target_detections=len(raw_detections),
                        lock_status=selection.status,
                        visual_backend=visual_backend,
                        position_source=selection.target.position_source if selection.target else "none",
                        position_valid=selection.target is not None,
                        position_invalid_reason=(
                            "" if selection.target is not None else
                            "ambiguous_no_unique_target" if selection.status == "ambiguous" else
                            "semantic_invalid" if active_position_reason == "semantic_invalid" else
                            "detector_anchor_stale" if active_position_reason == "detector_anchor_stale" else
                            "manual_active_position_unavailable" if selection.manual_active_lock else
                            selection.status),
                        visual_support_age_s=(selection.target.visual_support_age_s
                                              if selection.target else None),
                        position_age_s=selection.target.age_s if selection.target else None,
                        detection_age_s=selection.target.age_s if selection.target else None,
                                                                               
                                                                                  
                        detector_track_id=None,
                        detector_submitted=submitted,
                        detector_submit_reason=submit_reason,
                        detector_request_age_ms=request_age_ms,
                        detector_busy=detection_worker.stats()["busy"] if detection_worker else False,
                        detector_accepted=detector_accepted,
                        detector_result_age_ms=result_age_ms,
                        detector_search_roi=json.dumps(
                            result.context.get("search_roi") if result is not None else None),
                        detector_full_frame=(result.context.get("search_roi") is None
                                             if result is not None else
                                             True if detector_ran else None),
                        last_semantic_confirmed_at=(active_control_view.last_semantic_confirmed_at
                                                    if active_control_view else None),
                        semantic_age_s=(active_control_view.semantic_age_s
                                        if active_control_view else None),
                        semantic_state=(active_control_view.semantic_state
                                        if active_control_view else "none"),
                        semantic_reason=(active_control_view.semantic_reason
                                         if active_control_view else "no_active_view"),
                        full_frame_unmatched_count=(active_control_view.full_frame_unmatched_count
                                                    if active_control_view else None),
                        semantic_transition=semantic_transition,
                        association_reason=association_result.get("reason", "not_run"),
                        association_stage=json.dumps({
                            str(row["source_index"]): row.get("association_stage", "unmatched")
                            for row in detection_track_rows}, ensure_ascii=False),
                        mot_high_score_candidates=replay_diag.get("high_score_count", 0),
                        mot_low_score_candidates=replay_diag.get("low_score_count", 0),
                        mot_high_score_matches=replay_diag.get("high_score_matches", 0),
                        mot_low_score_matches=replay_diag.get("low_score_matches", 0),
                        mot_low_score_unmatched=replay_diag.get("low_score_unmatched", 0),
                        association_candidates=json.dumps(association_diag.get("association_costs", {}), ensure_ascii=False),
                        identity_recovered=False,
                        position_alignment="source_time_replay" if detector_accepted else "kalman_prediction",
                        visual_update_ok=bool(lk_diag.get("lk_visual_used", False)),
                        detector_source_sequence=source_sequence if detector_ran else None,
                        recovery_pending_count=selection.pending_track_id is not None,
                        visual_failure_reason=lk_diag.get("lk_reject_reason", ""),
                        visual_point_count=lk_diag.get("visual_point_count"),
                        lk_fwd_ok=lk_diag.get("lk_fwd_ok"),
                        lk_bwd_ok=lk_diag.get("lk_bwd_ok"),
                        lk_roundtrip_ok=lk_diag.get("lk_roundtrip_ok"),
                        lk_visual_used=lk_diag.get("lk_visual_used", False),
                        lk_ransac_reject_reason=lk_diag.get("lk_ransac_reject_reason", "not_run"),
                        lk_reject_reason=lk_diag.get("lk_reject_reason"),
                        lk_gate_rho=lk_diag.get("lk_gate_rho"),
                        lk_offset_ratio=lk_diag.get("lk_offset_ratio"),
                        lk_center_ratio=lk_diag.get("lk_center_ratio"),
                        lk_velocity_ratio=lk_diag.get("lk_velocity_ratio"),
                        lk_r_scale=lk_diag.get("lk_r_scale"),
                        lk_centroid_offset_ratio=lk_diag.get("lk_centroid_offset_ratio"),
                        lk_center_innovation_ratio=lk_diag.get("lk_center_innovation_ratio"),
                        lk_velocity_discrepancy_ratio=lk_diag.get("lk_velocity_discrepancy_ratio"),
                        visual_tracking_ms=lk_diag.get("visual_tracking_ms", 0.0),
                        visual_init_ms=lk_diag.get("visual_init_ms", 0.0),
                        visual_inlier_count=lk_diag.get("visual_inlier_count"),
                        visual_inlier_ratio=lk_diag.get("visual_inlier_ratio"),
                        visual_reprojection_error=lk_diag.get("visual_reprojection_error"),
                        visual_fb_error=lk_diag.get("visual_fb_error"),
                        visual_scale=lk_diag.get("visual_scale"),
                        visual_rotation_deg=lk_diag.get("visual_rotation_deg"),
                        visual_roi_area=lk_diag.get("visual_roi_area"),
                        alignment_ms=replay_diag.get("replay_ms", 0.0),
                        alignment_steps=replay_diag.get("replay_depth", 0),
                        alignment_reason=replay_diag.get("reason", ""),
                        alignment_method=replay_diag.get("replay_scope", "none"),
                        alignment_budget_exceeded=replay_diag.get("reason") == "replay_budget_exceeded",
                        fusion_uncertainty_px=(math.sqrt(max(selection.target.covariance[:2]))
                                               if selection.target else None),
                        fusion_observed_at=selection.target.last_detection_at if selection.target else None,
                        fusion_provenance="kalman" if selection.target else "none",
                        template_ready=lk_diag.get("template_ready", False),
                        template_verified=lk_diag.get("template_verified", False),
                        template_score=lk_diag.get("template_score"),
                        template_peak_margin=lk_diag.get("template_peak_margin"),
                        template_search_ms=lk_diag.get("template_search_ms", 0.0),
                        template_status=lk_diag.get("template_status", "disabled"),
                        template_confirmations=lk_diag.get("template_confirmations", 0),
                        template_seed_reject_reason=lk_diag.get("template_seed_reject_reason"),
                        template_anchor_score=lk_diag.get("template_anchor_score"),
                        template_best_score=lk_diag.get("template_best_score"),
                        template_peak_margin_diagnostic=lk_diag.get("template_peak_margin_diagnostic"),
                        template_timeout=lk_diag.get("template_timeout", False),
                        template_support_expired=lk_diag.get("template_support_expired", False),
                        raw_detection_candidates=json.dumps(detection_track_rows, ensure_ascii=False),
                        active_track_id=selection.active_track_id,
                        active_epoch=selection.active_epoch,
                        manual_active_lock=selection.manual_active_lock,
                        ambiguous_duration_s=selection.ambiguous_duration_s,
                        manual_acquire_result=selection.manual_request_result,
                        track_pool_full=capacity_log["full"],
                        track_capacity_event=capacity_log["event"],
                        track_capacity_rejections=capacity_log["rejected_candidates"]
                            if capacity_log["event"] else None,
                        track_count=len(views),
                        track_summaries=json.dumps([
                            dict(track_id=view.track_id, status=view.status, hits=view.hits,
                                 identity_confirmed=view.identity_confirmed,
                                 first_detection_sequence=view.first_detection_sequence,
                                 misses=view.misses, age_s=view.age_s, bbox=list(view.bbox),
                                 semantic_age_s=view.semantic_age_s,
                                 semantic_state=view.semantic_state,
                                 full_frame_unmatched_count=view.full_frame_unmatched_count,
                                 visual_support_age_s=view.visual_support_age_s,
                                 last_visual_support_at=view.last_visual_support_at,
                                 confidence=view.confidence, velocity=list(view.velocity),
                                 matched_detection_index=matched_detection_by_track.get(view.track_id),
                                 matched=view.track_id in matched_detection_by_track)
                            for view in views], ensure_ascii=False),
                        active_frontend_binding=json.dumps(
                            (active_frontend.summary().get("last_bind")
                             if active_frontend is not None else None), ensure_ascii=False),
                        selector_candidate_diagnostics=json.dumps(
                            list(selection.candidate_diagnostics), ensure_ascii=False),
                        association_assignments=json.dumps(assigned_ids, ensure_ascii=False),
                        association_detection_summaries=json.dumps(association_summaries, ensure_ascii=False),
                        association_costs=json.dumps(association_diag.get("association_costs", {}), ensure_ascii=False),
                        association_rejections=json.dumps(association_rejections_for_log, ensure_ascii=False),
                        association_pairwise_costs=json.dumps(
                            association_diag.get("association_pairwise_costs", {}), ensure_ascii=False)
                            if args.mot_debug_association_matrix else "",
                        association_pairwise_rejections=json.dumps(
                            association_diag.get("association_pairwise_rejections", {}), ensure_ascii=False)
                            if args.mot_debug_association_matrix else "",
                        pending_switch_id=selection.pending_track_id,
                        pending_switch_confirmations=selection.pending_confirmations,
                        replay_depth=replay_diag.get("replay_depth", 0),
                        replay_scope=replay_diag.get("replay_scope", "none"),
                        replay_ms=replay_diag.get("replay_ms", 0.0),
                        replay_reason=replay_diag.get("reason", "not_run"),
                        replay_source_sequence=replay_diag.get("source_sequence"),
                        replay_source_timestamp=replay_diag.get("source_timestamp"),
                        replay_source_age_s=replay_diag.get("source_age_s"),
                        replay_source_age_exceeds_history_window=replay_diag.get(
                            "source_age_exceeds_history_window"),
                        replay_source_timestamp_before_oldest_history=replay_diag.get(
                            "source_timestamp_before_oldest_history"),
                        replay_source_snapshot_missing_classification=replay_diag.get(
                            "source_snapshot_missing_classification"),
                        replay_source_snapshot_missing_causes=json.dumps(
                            replay_diag.get("source_snapshot_missing_causes", []), ensure_ascii=False),
                        replay_source_time_newer_track_ids=json.dumps(
                            replay_diag.get("source_time_newer_track_ids", []), ensure_ascii=False),
                        replay_source_before_oldest_boundary=replay_diag.get(
                            "source_before_oldest_boundary"),
                        replay_source_sequence_ahead_of_latest=replay_diag.get(
                            "source_sequence_ahead_of_latest"),
                        replay_source_boundary_timestamp=replay_diag.get("source_boundary_timestamp"),
                        replay_history_oldest_sequence=replay_diag.get("history_oldest_sequence"),
                        replay_history_latest_sequence=replay_diag.get("history_latest_sequence"),
                        replay_history_oldest_timestamp=replay_diag.get("history_oldest_timestamp"),
                        replay_history_latest_timestamp=replay_diag.get("history_latest_timestamp"),
                        replay_history_window_s=replay_diag.get("history_window_s"),
                        replay_source_candidate_count=replay_diag.get("source_candidate_count"),
                        replay_track_match_attempted=replay_diag.get("track_match_attempted"),
                        replay_source_time_track_ids=json.dumps(
                            replay_diag.get("source_time_track_ids", []), ensure_ascii=False),
                        replay_tombstone_track_ids=json.dumps(
                            replay_diag.get("tombstone_filtered_track_ids", []), ensure_ascii=False),
                        candidate_limited_count=replay_diag.get("limited_candidates", 0),
                        kf_covariance=json.dumps(list(selection.target.covariance) if selection.target else None),
                        stream_epoch=stream_epoch,
                        detector_poll_ms=detector_poll_ms,
                        detector_submit_ms=detector_submit_ms)
                elif visual_lock is not None:
                    submitted = False
                    submit_reason, request_age_ms = 'synchronous', None
                    poll_started = time.perf_counter()
                    result = detection_worker.poll() if detection_worker is not None else None
                    detector_poll_ms = (time.perf_counter()-poll_started)*1000 if detection_worker else 0.
                    if detection_worker is not None:
                        submit_reason = 'not_due'
                                                                                  
                                                                               
                                                                            
                        if result is None and visual_lock.detection_due(packet.received_at):
                            submitted, submit_reason, request_age_ms, detector_submit_ms = submit_detection_frame(frame, packet)
                        locked = visual_lock.update(frame, packet.received_at, detection=result)
                        if not submitted and visual_lock.detection_due(packet.received_at):
                            submitted, submit_reason, request_age_ms, detector_submit_ms = submit_detection_frame(frame, packet)
                    else:
                        locked = visual_lock.update(frame, packet.received_at, lambda: detect_targets(frame))
                    inference_ms, tracks = locked.inference_ms, locked.tracks
                    candidates = [locked.observation] if locked.observation is not None else []
                    measured = candidates if (candidates and locked.observation.position_source != "prediction") else []
                    if locked.status == "ambiguous":
                        measured = locked.raw_detections
                                                                                              
                    detections = candidates if detection_worker is not None else candidates + [
                        item for item in locked.raw_detections if not candidates
                        or bbox_iou(item.bbox, locked.observation.bbox) < 0.5]
                    lock_data = dict(
                        detector_ran=locked.detector_ran, raw_target_detections=len(locked.raw_detections),
                        lock_status=locked.status, visual_backend=visual_backend,
                        position_source=locked.observation.position_source if candidates else "none",
                        position_age_s=locked.position_age_s, detection_age_s=locked.detection_age_s,
                        detector_track_id=locked.detector_track_id,
                        detector_submitted=submitted,
                        detector_submit_reason=submit_reason, detector_request_age_ms=request_age_ms,
                        detector_busy=detection_worker.stats()["busy"] if detection_worker else False,
                        detector_accepted=locked.detector_accepted,
                        detector_result_age_ms=locked.detector_result_age_ms,
                        association_reason=locked.association_reason,
                        association_candidates=json.dumps(locked.association_candidates, ensure_ascii=False),
                        identity_recovered=locked.recovered, position_alignment=locked.position_alignment,
                        visual_update_ok=locked.visual_update_ok,
                        detector_source_sequence=result.sequence if result is not None else None,
                        recovery_pending_count=locked.recovery_pending_count,
                        visual_failure_reason=locked.visual_failure_reason,
                        visual_point_count=locked.visual_point_count,
                        lk_fwd_ok=locked.lk_fwd_ok,
                        lk_bwd_ok=locked.lk_bwd_ok,
                        lk_roundtrip_ok=locked.lk_roundtrip_ok,
                        lk_visual_used=locked.lk_visual_used,
                        lk_ransac_reject_reason=locked.lk_ransac_reject_reason,
                    )
                    for field in ('visual_tracking_ms', 'visual_init_ms', 'alignment_ms', 'history_store_ms',
                                  'visual_inlier_count', 'visual_inlier_ratio', 'visual_reprojection_error',
                                  'visual_fb_error', 'visual_scale', 'visual_rotation_deg', 'visual_roi_area',
                                  'alignment_steps', 'alignment_reason', 'alignment_method', 'alignment_budget_exceeded'):
                        lock_data[field] = getattr(locked, field)
                    lock_data.update(detector_reanchor_pending_count=locked.detector_reanchor_pending_count,
                                     detector_reanchored=locked.detector_reanchored)
                    lock_data.update(fusion_uncertainty_px=getattr(locked,'fusion_uncertainty_px',None),
                                     fusion_observed_at=getattr(locked,'fusion_observed_at',None),
                                     fusion_provenance=getattr(locked,'fusion_provenance','legacy'))
                    for field in ('template_ready','template_verified','template_score','template_peak_margin',
                                  'template_search_ms','template_status','template_confirmations',
                                  'flow_motion_hint_used','flow_guided_points'):
                        lock_data[field]=getattr(locked,field,None)
                    lock_data['raw_detection_candidates']=json.dumps([
                        dict(bbox=list(item.bbox),confidence=float(item.confidence),class_id=int(item.class_id))
                        for item in locked.raw_detections],ensure_ascii=False)
                else:
                    detections, tracks, inference_ms = detector.detect(frame)
                    candidates = [det for det in detections if args.target_class is None or det.class_id == args.target_class]
                    measured = candidates
                    lock_data = dict(detector_ran=True, raw_target_detections=len(candidates),
                                     lock_status="disabled", position_source="detector" if candidates else "none",
                                     visual_backend=visual_backend)
                vision_done = time.perf_counter()
                state = scene_metrics.observe(measured, width, height, time.perf_counter())
                if mot_tracker is None:
                    control_started = time.perf_counter()
                if control is not None and mot_tracker is None:
                    control_state = control.update(
                        candidates, width, height,
                        time.perf_counter(),
                        observed_at=packet.received_at,
                        selected_track_id=None)
                if mot_tracker is None:
                    control_done = time.perf_counter()
                if (mot_tracker is not None and not args.mot_show_all_tracks
                        and not args.mot_click_select):
                    detections = (candidates if control is None or control_state["control_position_valid"]
                                  else [])
                overlay_started = time.perf_counter()
                detector.draw(frame, detections, control_state)
                tracking_note = None
                if mot_tracker is not None:
                    tracking_note = "MOT tracks:%d | active:%s | control:%s | %s%s" % (
                        len(views), selection.active_track_id or "none",
                        ("off" if control is None else
                         "on" if control_state["control_position_valid"] else "hold"),
                        selection.status.upper(),
                        (" %.1fs" % selection.ambiguous_duration_s)
                        if selection.status == "ambiguous" else "")
                elif visual_lock is not None and not candidates:
                    if locked.recovery_pending_count:
                        tracking_note = 'RECOVERING: %d/%d confirmations' % (
                            locked.recovery_pending_count, args.recovery_confirmations)
                    elif locked.detector_ran and locked.raw_detections:
                        tracking_note = 'CANDIDATE FOUND: confirming current position'
                    else:
                        tracking_note = 'SEARCHING: detection active'
                overlay(cv2, frame, state, frame_metrics.last_state, inference_ms, args.deadzone,
                        control_state, show_stats=not args.hide_stats, tracking_note=tracking_note)
                if args.mot_click_select:
                    cv2.putText(frame, "CLICK CONFIRMED TRACK TO LOCK | %s" % (
                        click_state.get("click_feedback", "ready")),
                        (8, max(18, height - 30)), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (0, 255, 255), 1, cv2.LINE_AA)
                if video_frame_index is not None:
                    delta_label = "-" if seq_delta is None else str(seq_delta)
                    cv2.putText(frame, "F:%d SRC:%d D:%s" % (
                        video_frame_index, packet.sequence, delta_label),
                        (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
                overlay_done = time.perf_counter()
                if not args.no_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(log_dir / "output_sort.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create output video; try --no-video")
                    writer.write(frame)
                video_done = time.perf_counter()
                key = -1
                if not args.no_display:
                    view = fit_display(cv2, frame, args.display_max_width, args.display_max_height)
                    view_h, view_w = view.shape[:2]
                    geometry = {"width": view_w, "height": view_h,
                                "center_x": view_w // 2, "center_y": view_h // 2}
                    if geometry != display_geometry:
                        display_geometry = geometry
                        print("Display image: width=%d height=%d center=(%d,%d)" % (
                            view_w, view_h, view_w // 2, view_h // 2), flush=True)
                    cv2.imshow("Detection metrics", view)
                    if args.mot_click_select:
                        click_state["scale_x"] = float(view_w) / float(width)
                        click_state["scale_y"] = float(view_h) / float(height)
                        click_state["left_button_event"] = cv2.EVENT_LBUTTONDOWN
                        if not mouse_callback_registered:
                            cv2.setMouseCallback("Detection metrics", mot_click_callback, click_state)
                            mouse_callback_registered = True
                    key = cv2.waitKey(1) & 0xFF
                completed = time.perf_counter()
                timing = frame_metrics.record(started, completed, inference_ms)
                timing.update(
                    capture_ms=(captured_at - started) * 1000,
                    vision_other_ms=max(0.0, ((control_started if mot_tracker is not None
                                               else vision_done) - captured_at) * 1000
                                        - ((inference_ms or 0.0) if detection_worker is None else 0.0)),
                    control_ms=(control_done - control_started) * 1000,
                    control_decision_delay_ms=(max(0.0, control_started - control_at) * 1000
                                               if mot_tracker is not None else None),
                    output_ms=(completed - control_done) * 1000,
                    detector_poll_ms=detector_poll_ms, detector_submit_ms=detector_submit_ms,
                    overlay_ms=(overlay_done-overlay_started)*1000,
                    video_write_ms=(video_done-overlay_done)*1000 if not args.no_video else 0.,
                    display_ms=(completed-video_done)*1000 if not args.no_display else 0.,
                    frame_sequence=packet.sequence,
                    seq_delta=seq_delta,
                    video_frame_index=video_frame_index,
                    frames_overwritten=frame_reader.stats()["frames_overwritten"] if frame_reader else 0,
                    source_read_ms=(packet.received_at - packet.read_started_at) * 1000,
                    frame_age_ms=max(0.0, (control_started - packet.received_at) * 1000),
                    display_frame_age_ms=max(0.0, (completed - packet.received_at) * 1000),
                )
                csv_log.write(dict(timing, **state, **control_state, **lock_data,
                                   wall_time=datetime.now().astimezone().isoformat(timespec="milliseconds")))
                if completed - last_print >= 1.0:
                    print(
                        "frame=%d infer=%sms frame_time=%sms fps=%s ex=%s ey=%s err=%s "
                        "center=%ss lost=%d id_changes=%d status=%s ctrl=%s pan=%s tilt=%s limits=%s/%s "
                        "pan_reason=%s tilt_reason=%s age=%sms drop=%d "
                        "control_valid=%s control_ex=%s control_ey=%s pan_step=%s tilt_step=%s pan_rate=%s tilt_rate=%s pid_sat=%s/%s ff=%s/%s vel_ready=%s "
                        "lock=%s pos=%s yolo=%s busy=%s accepted=%s assoc=%s recover=%s det=%s result_age=%sms align=%s pending=%s" % (
                            timing["frame_index"], number(inference_ms), number(timing["frame_ms"]),
                            number(timing["actual_fps"]), number(state["error_x"], 3),
                            number(state["error_y"], 3), number(state["error_norm"], 3),
                            number(state["last_center_time_s"], 3), state["loss_count"],
                            state["id_switch_count"], state["status"], control_state["control_status"],
                            number(control_state.get("pan_angle_deg"), 2), number(control_state.get("tilt_angle_deg"), 2),
                            control_state.get("pan_limited", False), control_state.get("tilt_limited", False),
                            control_state.get("pan_reason", "disabled"), control_state.get("tilt_reason", "disabled"),
                            number(timing["frame_age_ms"]), timing["frames_overwritten"],
                            control_state.get("control_position_valid", False),
                            number(control_state.get("control_error_x"), 3),
                            number(control_state.get("control_error_y"), 3),
                            number(control_state.get("pan_step_deg"), 3),
                            number(control_state.get("tilt_step_deg"), 3),
                            number(control_state.get("pan_rate_deg_s"), 2),
                            number(control_state.get("tilt_rate_deg_s"), 2),
                            control_state.get("pan_pid_saturated", False),
                            control_state.get("tilt_pid_saturated", False),
                            number(control_state.get("pan_feedforward_deg_s"), 2),
                            number(control_state.get("tilt_feedforward_deg_s"), 2),
                            control_state.get("velocity_ready", False),
                            lock_data["lock_status"], lock_data["position_source"], lock_data["detector_ran"],
                            lock_data.get("detector_busy", False), lock_data.get("detector_accepted", False),
                            lock_data.get("association_reason", "detection_only"),
                            lock_data.get("identity_recovered", False),
                            lock_data["raw_target_detections"] if lock_data["detector_ran"] else "N/A",
                            number(lock_data.get("detector_result_age_ms")),
                            lock_data.get("position_alignment", "current_frame"),
                            lock_data.get("recovery_pending_count", 0)),
                        flush=True,
                    )
                    if visual_lock is not None:
                        progress = visual_lock.summary()
                        latest = progress['last_detection_result'] or {}
                        worker = detection_worker.stats() if detection_worker is not None else {}
                        print(
                            "detect_total=%d hit_total=%d accepted_total=%d worker_done=%s busy=%s "
                            "last_det=%s last_assoc=%s last_align=%s last_align_reason=%s last_result_age=%sms reanchor_total=%d reanchor_pending=%d" % (
                                progress['detector_calls'], progress['detector_hit_frames'],
                                progress['confirmed_detection_frames'], worker.get('completed', 'sync'),
                                worker.get('busy', False), latest.get('target_count', 'N/A'),
                                latest.get('association_reason', 'no_result_yet'),
                                latest.get('position_alignment', 'none'),
                                latest.get('alignment_reason') or 'none',
                                number(latest.get('result_age_at_consumption_ms')),
                                progress['detector_reanchors'], progress['detector_reanchor_pending_count']),
                            flush=True,
                        )
                    last_print = completed
                if key == ord("q"):
                    reason = "user_quit"
                    break
                if args.max_frames and frame_metrics.frame_count >= args.max_frames:
                    reason = "frame_limit"
                    break
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception as exc:
        reason, failure = "error", "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        if detection_worker is not None:
            detection_details["worker_stopped"] = detection_worker.close()
            detection_details.update(detection_worker.stats())
            if not detection_details["worker_stopped"]:
                print("Detector did not stop within 2 seconds; pending work was cancelled", flush=True)
            if detection_details.get("error") and failure is None:
                reason, failure = "error", "Detector failed: %s" % detection_details["error"]
        if frame_reader is not None:
            capture_details["reader_stopped"] = frame_reader.close()
            capture_details.update(frame_reader.stats())
            if not capture_details["reader_stopped"]:
                print("Camera driver read is still blocked; reader did not stop within 2 seconds", flush=True)
        elif capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if cv2 is not None and not args.no_display:
            cv2.destroyAllWindows()
        summary = {
            "version": VERSION,
            "control_algorithm": "PID+velocity_feedforward" if servo_config.velocity_feedforward else "PID",
            "ended_at": datetime.now().astimezone().isoformat(), "exit_reason": reason,
            "error": failure,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "timing": frame_metrics.summary() if frame_metrics is not None else {},
            "scene": scene_metrics.summary(),
            "servo": control_state,
            "servo_mapping": {
                "actuation_range_deg": servo_config.actuation_range,
                "pulse_min_us": servo_config.pulse_min, "pulse_max_us": servo_config.pulse_max,
            },
            "servo_test_commands": servo_test_commands,
            "capture": capture_details,
            "detection": detection_details,
            "model": model_details,
            "detector_debug": detector_debug_details,
            "visual_lock": dict(enabled=visual_lock is not None, backend=visual_backend,
                                flow_config=asdict(flow_config),
                                alignment_history=projector.stats() if callable(getattr(projector, 'stats', None)) else {},
                                **(visual_lock.summary() if visual_lock is not None else {})),
            "multi_target_tracking": dict(
                enabled=mot_tracker is not None,
                mode=mot_details.get("mode", "legacy_single_target"),
                tracker=mot_tracker.summary() if mot_tracker is not None else {},
                selector=target_selector.summary() if target_selector is not None else {},
                active_frontend=active_frontend.summary() if active_frontend is not None else {}),
            "detector_association_definition": (
            "MOT mode retains ONNX output rows down to mot_low_score_threshold. Rows at or above confThreshold are associated first; lower-score rows then associate to any unmatched track, including tentative tracks. Unmatched rows above mot_birth_confidence may create tentative IDs; IDs require distinct-source-frame confirmation before selection. Per-candidate diagnostics record score_stage, association_stage, assigned ID, and rejection reason. Candidates removed inside ONNX/NMS cannot be recovered." if mot_tracker is not None else
                "Legacy mode retains the configured confThreshold."),
            "manual_click_selection_definition": (
                "With mot_click_select enabled, clicking inside a confirmed/lost track queues request_acquire(track_id) for the main loop. Click coordinates are mapped from the fitted display to source-image coordinates; tentative tracks are not clickable. Ambiguous and invalid positions keep the gimbal on hold." if mot_tracker is not None and args.mot_click_select else
                "Disabled; pass --mot-click-select to enable operator selection."),
            "image_geometry": image_geometry, "display_geometry": display_geometry,
            "frame_age_definition": "Milliseconds since VideoCapture.read returned, NOT sensor/exposure-to-screen latency",
            "servo_angle_definition": "Commanded ServoKit angles, not measured shaft position; home angles, channels and directions are recorded in config",
            "pid_definition": "P/I/D terms and unsaturated output are rates before axis direction; signed rate is actual commanded step divided by elapsed seconds",
            "velocity_definition": "Optional local angular-rate estimate from image bearings plus time-aligned commanded camera pose; no measured shaft feedback; receive time is not exposure time",
            "id_switch_definition": ("MOT track IDs are source-time association identities, not ground-truth identity; active switches require selector confirmation"
                                     if mot_tracker is not None else
                                     (("Single-target continuity IDs; not ground-truth identity or MOT IDSW"
                                       if not args.legacy_fusion else "Single-track time-based SORT continuity IDs; not ground-truth identity or MOT IDSW")
                                      if visual_lock is not None else "Diagnostic ID changes of one unchanged physical target; not ground-truth MOT IDSW")),
            "scene_measurement_definition": ("MOT: statistics use only a current-time active detector observation; Kalman prediction and LK velocity updates are excluded from visible/centering measurements"
                                             if mot_tracker is not None else "With visual lock: detector and image-tracker positions are measurements; prediction is excluded from visible/centering statistics"),
            "inference_ms_definition": "Async: worker inference duration on the frame that consumes its result, not current-frame blocking time; mean covers consumed results only. Worker completed count includes unconsumed results. Sync: actual current-frame inference duration.",
            "detector_ran_definition": "Async: one completed result consumed this frame; use detector_submitted/busy/accepted to distinguish scheduling, execution and association. Sync: YOLO executed this frame.",
            "detector_search_roi_definition": "In live asynchronous MOT search or semantic-suspect state, alternating accepted YOLO submissions use the full source image or a semantically recent track-centered crop. Non-null detector_search_roi is the crop [left, top, right, bottom] in source coordinates; detector boxes are remapped before source-time association. Synchronous and legacy paths use full frames. ROI-negative results do not count as full-frame semantic misses.",
            "semantic_lease_definition": "Each MOT ID records its latest matched YOLO source time separately from LK motion support. At age>=mot_semantic_soft_age with mot_semantic_misses distinct accepted full-frame unmatched results it becomes suspect; suspect alone never preempts the active ID. At age>=mot_semantic_hard_age it cannot control or renew LK, even if geometric flow remains valid. The active ID remains selected until its mot_retain_seconds deadline from its last YOLO match; during invalid position the servo holds. After expiry, automatic handoff requires exactly one eligible confirmed replacement; multiple replacements remain ambiguous. Frames.csv records semantic age/state/reason, full-frame miss count and transitions. Thresholds require Pi video validation.",
            "control_decision_delay_ms_definition": "MOT: wall-clock time from the timestamp shared by selector and PID until controller.update begins. The command is issued before association logging, overlay, video encoding and display. Null outside MOT mode.",
            "confirmed_detection_frames_definition": "Accepted detection results, including confirmations that keep the newer visual position; may overlap visual_frames.",
            "stage_timing_definition": "MOT replay_ms/alignment_ms covers source-time association replay; visual_tracking_ms covers active LK and RANSAC; LK, replay and selector run on the main tracking thread. overlay_ms covers all-track drawing and HUD only; output_ms covers all post-control diagnostics, overlay, video and display; display_ms includes resize, imshow and waitKey.",
            "lk_diagnostics_definition": "lk_fwd_ok and lk_bwd_ok count finite successful forward and reverse LK statuses separately; lk_roundtrip_ok also applies the forward-backward error gate. In MOT mode lk_visual_used means a quality-gated midpoint LK speed observation updated only vx,vy in the active track KF; the LK affine box is diagnostic and never a position measurement. R_L multipliers are 1x/2x/4x for rho<=1/2/3; rho>3 rejects the event. Initial soft limits are uncalibrated until labeled clips are collected.",
            "multi_target_definition": "The default MOT path disables detector-side SORT IDs. High-score YOLO candidates are associated first by gated Hungarian assignment; low-score candidates then associate to unmatched tracks, including tentative tracks. Any unmatched candidate at or above the 0.15 birth floor may create a tentative ID after duplicate-center and capacity checks; an ID remains non-controlling until mot_confirm_hits distinct source frames (default 2) confirm it. High-score threshold follows confThreshold (default 0.35). Duplicate-center rejection uses 0.5 times the same-class confirmed track's predicted box short side only when max(sigma_x,sigma_y) is at most 0.5 times that short side; otherwise the guard is skipped. Per-track state is [cx,cy,logw,logh,vx,vy]; YOLO updates center/size and LK updates velocity only at frame-interval midpoints. Qualified LK may refresh the 0.35 second motion-support horizon but cannot refresh YOLO semantic identity or the 3 second track-retention horizon. Delayed detections retain source sequence/time and use bounded source-time replay; missing source snapshots are rejected, never approximated. All tracks continue updating; only one active track reaches LK/template and PID. Detection list order does not determine assignment.",
            "identity_confirmation_definition": "identity_confirmed is true when the replay-consistent track state contains at least mot_confirm_hits distinct detector source sequences (default 2). This applies equally to high- and low-score tentative births; a one-frame candidate never reaches control. Automatic selection requires an identity-confirmed same-class track with fresh position and acceptable covariance. Initial selection is deterministic; after active-ID expiry, automatic handoff occurs only with one eligible confirmed replacement, while multiple replacements remain ambiguous. The active ID remains locked for mot_retain_seconds (default 3.0 s) after its last YOLO match; the servo predicts for at most mot_prediction_horizon (default 0.35 s), then holds while waiting for same-ID recovery or expiry. mot_switch_confirmations remains accepted for CLI compatibility but does not add another confirmation stage.",
            "selector_candidate_diagnostics_definition": "Each frames.csv row lists selector candidates with track_id, lifecycle status, hit count, identity_confirmed, detector age, visual-support age, eligibility, and rejection reason (identity_unconfirmed, lifecycle_ineligible, position_stale, invalid_state, position_uncertain, or eligible).",
            "detector_id_definition": "In multi_target_kf mode detector_track_id is null because detector-side SORT is disabled; per-candidate MOT IDs and best/second match summaries are in frames.csv raw_detection_candidates and association_detection_summaries. active_track_id is the control selector's chosen MOT track, not a detector SORT ID or real-world identity.",
            "ambiguous_definition": "A replacement must be identity-confirmed with mot_confirm_hits distinct detector source frames (default 2). The active ID remains locked through its mot_retain_seconds window even after becoming suspect/invalid; position prediction is capped by mot_prediction_horizon (default 0.35 s), then the servo holds. Once the active ID expires, exactly one eligible confirmed replacement may be selected automatically; multiple candidates yield ambiguous and hold until only one remains or an operator selects an ID. Explicit request_acquire(track_id) pins the requested confirmed/lost identity. Same-ID YOLO recovery restores tracking during retention. Tentative IDs never control the gimbal.",
            "detector_jump_gate_definition": "Before Hungarian association, a detector candidate is rejected as a match to an already-confirmed track when center displacement from the last detector anchor is at least mot_jump_center_ratio (default 2.0) target short sides and its speed is at least mot_jump_speed_short_sides (default 6.0) short sides/s. The unmatched candidate can create a tentative ID if birth and capacity gates allow; it cannot move the confirmed track or control the gimbal until independently confirmed. This supplements the existing Kalman velocity guard.",
            "active_frontend_binding_definition": "active_frontend_binding records the track ID, current processed frame sequence/time, current filter box, and bind reason whenever the LK frontend is initialized or rebound. Active-ID handoff initializes LK from the current frame and current-time track box; prior track points and grayscale cache are discarded.",
            "manual_lock_recovery_definition": "A valid same-ID YOLO association during the 3.0-second retention window restores semantic validity and tracking; quality-gated LK can support motion only before the 2.0-second semantic hard deadline. The active ID does not change, the selector does not restart acquisition confirmation, and the controller preserves that identity's PID integral/EMA/velocity state through later prediction/measured-source transitions; this preservation marker clears only on active-ID change or manual-lock release. During position-invalid hold, commanded servo angles stay at their last values because the program issues no new servo writes; it does not send a zero-rate or home command.",
            "birth_capacity_precedence_definition": "For an unmatched detection above birth confidence, capacity preflight runs before the sigma-based duplicate-birth guard. If the pool has no slot and no lifecycle-expired track can be pruned, reject with track_capacity and record birth_guard_evaluated=false / birth_guard_skip_reason=capacity_preflight; sigma uncertainty can never bypass the cap. Only when capacity is available (including a safely prunable expired slot) is the duplicate guard evaluated; high sigma then means skip that guard and allow the candidate to continue through birth/capacity handling.",
            "active_switch_reset_definition": "On a real active-ID switch, the controller clears both PID integral terms, derivative/previous-error history, EMA-smoothed errors, target velocity feed-forward history, active LK tracker including its point list and previous grayscale-frame cache, template bank and template confirmation count. It preserves the current commanded pan/tilt positions and command-pose history; it does not home the servos or write a zero-rate command.",
            "lost_after_calibration_definition": "mot_lost_after defaults to 0.25 seconds only as a provisional starting point. Calibrate it from consecutive positive deltas of replay_source_timestamp on consumed YOLO result rows: P90 interval times 1.5 is the target lower bound. Keep lost_after below prediction_horizon; if 1.5*P90 is not below 0.35 seconds, there is no feasible overlap without reducing detector interval or revisiting the horizon. lost_after (lost-state transition), mot_tentative_timeout (default 1.0 seconds), mot_retain_seconds (default 3.0 seconds), and prediction_horizon (default 0.35 seconds without valid YOLO/LK support) are separate timers.",
            "track_capacity_definition": "The active pool is capped at mot_max_tracks. Tentative tracks expire after mot_tentative_timeout seconds; confirmed/lost tracks remain available for reassociation up to mot_retain_seconds. Capacity handling removes only lifecycle-expired tracks and otherwise rejects a new track with track_capacity. The pool-full transition is logged once and sustained-full summaries at 5-second intervals. Replay rejection deduplication uses (source_sequence, track_or_candidate_key, reason), where candidate index disambiguates multiple detections in a frame. Source-time snapshots remain available for mot_history_seconds; a retired ID may aid historical replay only before its tombstone boundary.",
            "source_snapshot_definition": "A delayed detector result whose source sequence has no retained frame-boundary snapshot is rejected with source_snapshot_missing. CSV records source sequence/time/age, age>history-window and timestamp<oldest-snapshot flags, missing classification/causes, oldest/latest retained sequence/time, configured history window, candidate count, and that source-time matching was not attempted; source-time track IDs are empty because no source snapshot exists. The multi-label source_snapshot_missing_causes directly reports history_window_too_short when age>history_window, detection_older_than_oldest_snapshot when source_timestamp<oldest_snapshot, and track_born_after_source_timestamp with same-class track IDs whose first_detection_at>source_timestamp; labels may co-occur. These newer tracks are diagnostic only and are never matched or updated from current state. Tombstone-filtered IDs on an accepted replay are logged separately. No current-state or nearest-snapshot approximation is used.",
            "tombstone_definition": "Tombstone retention defaults to 1.0 second and must be at least the replay history window (default 0.5 second). The CLI rejects a shorter retention; referenced snapshots can keep a tombstone longer.",
            "sequence_video_alignment_definition": "seq_delta is current source frame sequence minus previous processed source sequence (first frame null); video_frame_index is the 1-based processed/output video frame ordinal when video saving is enabled and aligns with CSV frame_index. Encoded output uses constant FPS and does not preserve camera capture timing; use wall_time/elapsed_s and frame_sequence for temporal analysis.",
            "fusion_definition": ("Per-track Kalman state: YOLO is the center/size observation, quality-gated LK is a midpoint velocity observation, and delayed YOLO is replayed at source time" if mot_tracker is not None else
                ("Legacy affine-box fusion" if args.legacy_fusion else
                 "Detector-owned dimensions; relative center displacement replay; uncertainty_px is a conservative engineering error budget, not calibrated covariance; uncovered propagation remains prediction unless separately validated current-image evidence closes the gap")),
            "frame_ms_definition": "Capture through servo API calls, overlay, video write and display; excludes current CSV/console emission and physical servo completion",
            "control_position_valid_definition": "True only when the pan/tilt controller has accepted one fresh, valid target position for automatic control. It is distinct from selector position_valid, which records whether the active tracker supplied a valid position before controller-side target gating.",
            "fps_definition": "Rolling completion-to-completion throughput, including intervening logging; video input is not paced to source FPS",
        }
        with (log_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        print(json.dumps(summary["timing"], ensure_ascii=False))
        print(json.dumps(summary["scene"], ensure_ascii=False))
        print("Summary: %s" % (log_dir / "summary.json").resolve())
    if failure is not None:
        raise RuntimeError(failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
