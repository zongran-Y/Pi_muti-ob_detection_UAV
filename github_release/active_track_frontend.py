"""LK/template front end bound to the selector's one active track ID."""

import math
import time

import numpy as np


class ActiveTrackFrontend:
    """Produce gated LK midpoint velocity observations, never LK box updates."""

    def __init__(self, tracker_factory, template_bank=None, offset_soft_limit=1.0,
                 center_soft_limit=.75, velocity_soft_limit=1.5,
                 base_velocity_sigma_px_s=18.0):
        values = (offset_soft_limit, center_soft_limit, velocity_soft_limit,
                  base_velocity_sigma_px_s)
        if (not all(math.isfinite(value) for value in values)
                or min(values) <= 0):
            raise ValueError("LK gate values must be finite and positive")
        self.factory = tracker_factory
        self.template_bank = template_bank
        self.offset_soft_limit = float(offset_soft_limit)
        self.center_soft_limit = float(center_soft_limit)
        self.velocity_soft_limit = float(velocity_soft_limit)
        self.base_velocity_sigma_px_s = float(base_velocity_sigma_px_s)
        self._supported = (tracker_factory is not None
                           and getattr(tracker_factory, "method", None) == "roi_lk_ransac")
        self.reset()

    def reset(self, reset_bank=True):
        """Drop LK points and previous-gray cache, active-ID state, and optional templates."""
        self.track_id = None
        self.tracker = None
        self.last_timestamp = None
        self.last_sequence = None
        self.last_detector_at = None
        self.last_frame_shape = None
        self.last_box = None
        self.last_bind = None
        self.last_diagnostics = self._empty_diagnostics()
        if self.template_bank is not None:
            if reset_bank:
                self.template_bank.reset()
            self.template_recovery = None
        else:
            self.template_recovery = None

    @staticmethod
    def _empty_diagnostics(reason="not_run"):
        return dict(lk_fwd_ok=None, lk_bwd_ok=None, lk_roundtrip_ok=None,
                    lk_visual_used=False, lk_ransac_reject_reason=reason,
                    lk_reject_reason="", lk_gate_rho=None,
                    lk_offset_ratio=None, lk_center_ratio=None,
                    lk_velocity_ratio=None, lk_r_scale=None,
                    lk_centroid_offset_ratio=None, lk_center_innovation_ratio=None,
                    lk_velocity_discrepancy_ratio=None,
                    visual_point_count=0, visual_inlier_count=0,
                    visual_inlier_ratio=None, visual_reprojection_error=None,
                    visual_fb_error=None, visual_scale=None,
                    visual_rotation_deg=None, visual_roi_area=None,
                    template_ready=False, template_verified=False,
                    template_score=None, template_peak_margin=None,
                    template_search_ms=0.0, template_status="disabled",
                    template_confirmations=0, template_seed_reject_reason="disabled",
                    template_anchor_score=None, template_best_score=None,
                    template_peak_margin_diagnostic=None, template_timeout=False,
                    template_support_expired=False)

    def _template_diagnostics(self, diagnostics):
        if self.template_bank is None:
            return diagnostics
        bank = self.template_bank.diagnostics
        recovery = self.template_recovery
        diagnostics.update(
            template_ready=bool(self.template_bank.ready),
            template_status=recovery.status if recovery is not None else "unseeded",
            template_confirmations=recovery.streak if recovery is not None else 0,
            template_seed_reject_reason=bank.get("seed_reject_reason"),
            template_anchor_score=bank.get("anchor_score"),
            template_best_score=bank.get("best_score"),
            template_peak_margin_diagnostic=bank.get("peak_margin"),
            template_timeout=bool(bank.get("search_timed_out", False)),
            template_support_expired=bool(recovery is not None
                                          and recovery.status == "semantic_support_expired"))
        return diagnostics

    def bind(self, frame, track_view, timestamp, sequence, active_epoch):
        identity = track_view.track_id if track_view is not None else None
        changed = identity != self.track_id
        bind_reason = "active_id_changed" if changed else "frontend_reinitialized"
        if changed:
            self.reset(reset_bank=True)
            self.track_id = identity
        if identity is None:
            return self.last_diagnostics
        self.last_timestamp = float(timestamp)
        self.last_sequence = int(sequence)
        self.last_frame_shape = tuple(frame.shape[:2])
        self.last_box = tuple(float(value) for value in track_view.bbox)
        self.last_detector_at = float(track_view.last_detection_at)
        self.active_epoch = int(active_epoch)
        self.last_bind = dict(track_id=str(identity), frame_sequence=int(sequence),
                              timestamp=float(timestamp), bbox=self.last_box,
                              reason=bind_reason)
        self.last_diagnostics = self._empty_diagnostics("not_run" if self._supported else "lk_backend_unavailable")
        if not self._supported:
            self._template_diagnostics(self.last_diagnostics)
            return self.last_diagnostics
        self.tracker = self.factory()
        started = time.perf_counter()
        try:
            ready = bool(self.tracker.init(frame, self.last_box))
        except Exception:
            ready = False
        self.last_diagnostics["visual_init_ms"] = (time.perf_counter() - started) * 1000.0
        if not ready:
            self.last_diagnostics["lk_ransac_reject_reason"] = getattr(
                self.tracker, "last_failure_reason", "tracker_init_failed")
            self.tracker = None
        if self.template_bank is not None:
            from template_recovery import TemplateRecovery
            self.template_recovery = TemplateRecovery(self.template_bank)
        self._template_diagnostics(self.last_diagnostics)
        return self.last_diagnostics

    def record_detector(self, frame, track_view, bbox, confidence, source_timestamp,
                        source_sequence, active_epoch, current_frame=False):
        if (self.template_bank is None or track_view is None
                or track_view.track_id != self.track_id):
            return dict(seed_reject_reason="not_active_track")
        self.last_detector_at = float(source_timestamp)
        changed = self.template_bank.consider(
            frame, bbox, confidence, source_timestamp,
            image_current=True, stable=track_view.status == "confirmed")
        diagnostics = dict(self.template_bank.diagnostics)
        if changed and self.template_recovery is not None:
            self.template_recovery.clear_confirmation("seeded")
                                                                                
        if current_frame and self._supported and self.tracker is not None:
            try:
                if not self.tracker.reanchor(frame, track_view.bbox):
                    self.tracker = self.factory()
                    if not self.tracker.init(frame, track_view.bbox):
                        self.tracker = None
            except Exception:
                self.tracker = None
            if self.tracker is not None:
                self.last_box = tuple(float(value) for value in track_view.bbox)
        return diagnostics

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) * .5, (box[1] + box[3]) * .5)

    def _predicted_box(self, track_view, dt):
        box = tuple(float(value) for value in track_view.bbox)
        vx, vy = track_view.velocity
        dx, dy = float(vx) * dt, float(vy) * dt
        return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy

    def _try_template_recovery(self, frame, reference, timestamp, track_view):
        if self.template_bank is None or self.template_recovery is None:
            return None
        match = self.template_recovery.check(
            frame, reference, timestamp, track_view.last_detection_at)
        bank = self.template_bank.diagnostics
        self.last_diagnostics.update(
            template_ready=bool(self.template_bank.ready),
            template_verified=match is not None,
            template_score=getattr(match, "score", bank.get("best_score")),
            template_peak_margin=getattr(match, "margin", bank.get("peak_margin")),
            template_search_ms=bank.get("elapsed_ms", 0.0),
            template_status=self.template_recovery.status,
            template_confirmations=self.template_recovery.streak,
            template_anchor_score=bank.get("anchor_score"),
            template_best_score=bank.get("best_score"),
            template_peak_margin_diagnostic=bank.get("peak_margin"),
            template_timeout=bank.get("reason") == "timeout",
            template_support_expired=self.template_recovery.status == "semantic_support_expired")
        if match is not None and self._supported:
                                                                          
                                                                          
            self.tracker = self.factory()
            try:
                if not self.tracker.init(frame, match.bbox):
                    self.tracker = None
            except Exception:
                self.tracker = None
            if self.tracker is not None:
                self.last_box = tuple(float(value) for value in match.bbox)
        return match

    def observe_interval(self, frame, timestamp, sequence, track_view, active_epoch):
        diagnostics = self._empty_diagnostics("not_run" if self._supported else "lk_backend_unavailable")
        semantic_deadline = getattr(track_view, "semantic_valid_until", None)
        if (track_view is not None and
                (getattr(track_view, "semantic_state", "verified") == "invalid"
                 or (semantic_deadline is not None and timestamp >= semantic_deadline))):
            diagnostics["lk_ransac_reject_reason"] = "semantic_expired"
            diagnostics["lk_reject_reason"] = "semantic_expired"
            self.tracker = None
            self.last_diagnostics = diagnostics
            return None, diagnostics
        if (track_view is None or track_view.track_id != self.track_id
                or self.tracker is None or self.last_timestamp is None
                or self.last_sequence is None):
            self.last_diagnostics = diagnostics
            return None, diagnostics
        timestamp, sequence = float(timestamp), int(sequence)
        t0, t1 = self.last_timestamp, timestamp
        dt = t1 - t0
        if (not math.isfinite(timestamp) or dt <= 0 or sequence <= self.last_sequence
                or tuple(frame.shape[:2]) != self.last_frame_shape):
            diagnostics["lk_ransac_reject_reason"] = "invalid_interval_or_frame_shape"
            self.last_diagnostics = diagnostics
            self.bind(frame, track_view, timestamp, sequence, active_epoch)
            return None, diagnostics
        predicted_box = self._predicted_box(track_view, dt)
        started = time.perf_counter()
        try:
            ok, lk_box = self.tracker.update(frame)
        except Exception as exc:
            ok, lk_box = False, None
            diagnostics["lk_ransac_reject_reason"] = "tracker_exception:%s" % type(exc).__name__
        tracking_ms = (time.perf_counter() - started) * 1000.0
        quality = dict(getattr(self.tracker, "diagnostics", {}) or {})
        diagnostics.update(
            lk_fwd_ok=quality.get("fwd_ok"),
            lk_bwd_ok=quality.get("bwd_ok"),
            lk_roundtrip_ok=quality.get("roundtrip_ok"),
            lk_ransac_reject_reason=quality.get("ransac_reject_reason",
                                                getattr(self.tracker, "last_failure_reason", "not_run")),
            visual_point_count=quality.get("consistent_points", 0),
            visual_inlier_count=quality.get("inlier_count", 0),
            visual_inlier_ratio=quality.get("inlier_ratio"),
            visual_reprojection_error=quality.get("reprojection_error"),
            visual_fb_error=quality.get("fb_error"),
            visual_scale=quality.get("scale"),
            visual_rotation_deg=quality.get("rotation_deg"),
            visual_roi_area=quality.get("roi_area"),
            visual_tracking_ms=tracking_ms)
        event = None
        if ok and lk_box is not None:
            old_center = self._center(self.last_box)
            new_center = self._center(lk_box)
            velocity = ((new_center[0] - old_center[0]) / dt,
                        (new_center[1] - old_center[1]) / dt)
            pred_center = self._center(predicted_box)
            short_side = max(2.0, min(self.last_box[2] - self.last_box[0],
                                      self.last_box[3] - self.last_box[1]))
            centroid = quality.get("inlier_centroid")
            if centroid is None or len(centroid) != 2:
                centroid = new_center
            centroid_offset = math.hypot(float(centroid[0]) - pred_center[0],
                                         float(centroid[1]) - pred_center[1]) / short_side
            center_innovation = math.hypot(new_center[0] - pred_center[0],
                                           new_center[1] - pred_center[1]) / short_side
            predicted_velocity = tuple(float(value) for value in track_view.velocity)
            delta_velocity = math.hypot(velocity[0] - predicted_velocity[0],
                                        velocity[1] - predicted_velocity[1])
            velocity_scale = max(math.hypot(*predicted_velocity),
                                 short_side / max(dt, 1e-3) * .25, 5.0)
            velocity_discrepancy = delta_velocity / velocity_scale
            ratios = (centroid_offset / self.offset_soft_limit,
                      center_innovation / self.center_soft_limit,
                      velocity_discrepancy / self.velocity_soft_limit)
            rho = max(ratios)
            diagnostics.update(lk_gate_rho=rho,
                               lk_offset_ratio=ratios[0],
                               lk_center_ratio=ratios[1],
                               lk_velocity_ratio=ratios[2],
                               lk_centroid_offset_ratio=centroid_offset,
                               lk_center_innovation_ratio=center_innovation,
                               lk_velocity_discrepancy_ratio=velocity_discrepancy)
            if not all(math.isfinite(value) for value in (*velocity, rho)):
                diagnostics["lk_reject_reason"] = "non_finite_quality"
            elif rho > 3.0:
                diagnostics["lk_reject_reason"] = "identity_motion_gate_rho_gt_3"
            else:
                scale = 1.0 if rho <= 1.0 else (2.0 if rho <= 2.0 else 4.0)
                fb_error = float(quality.get("fb_error") or 0.0)
                sigma = max(self.base_velocity_sigma_px_s,
                            short_side * .03 / max(dt, 1e-3),
                            2.0 * fb_error / max(dt, 1e-3))
                variance = sigma * sigma * scale
                R = np.diag((variance, variance))
                diagnostics.update(lk_visual_used=True, lk_r_scale=scale)
                event = dict(track_id=self.track_id,
                             sequence0=self.last_sequence, sequence1=sequence,
                             t0=t0, t1=t1, measurement_at=(t0 + t1) * .5,
                             z=velocity, R=R,
                             quality=dict(rho=rho, ratios=ratios,
                                          r_scale=scale,
                                          inlier_count=quality.get("inlier_count"),
                                          inlier_ratio=quality.get("inlier_ratio"),
                                          fb_error=quality.get("fb_error"),
                                          inlier_centroid=centroid,
                                          lk_bbox=tuple(float(value) for value in lk_box)),
                             active_epoch=int(active_epoch))
        else:
            diagnostics["lk_reject_reason"] = diagnostics.get("lk_ransac_reject_reason") or "lk_failed"

        next_frontend_box = predicted_box
        if event is not None:
            next_frontend_box = tuple(float(value) for value in lk_box)
        if event is None:
                                                                              
                                                                          
                                        
            self.tracker = self.factory()
            try:
                if not self.tracker.init(frame, predicted_box):
                    self.tracker = None
            except Exception:
                self.tracker = None
            match = self._try_template_recovery(frame, predicted_box, timestamp, track_view)
            if match is not None and self.tracker is not None:
                next_frontend_box = tuple(float(value) for value in match.bbox)
        self.last_timestamp = timestamp
        self.last_sequence = sequence
        self.last_frame_shape = tuple(frame.shape[:2])
        self.last_box = tuple(float(value) for value in next_frontend_box)
        self.last_detector_at = float(track_view.last_detection_at)
        self.active_epoch = int(active_epoch)
        self._template_diagnostics(diagnostics)
        self.last_diagnostics = diagnostics
        return event, diagnostics

    def summary(self):
        return dict(enabled=self._supported, backend=getattr(self.factory, "method", "disabled"),
                    active_track_id=self.track_id,
                    last_bind=dict(self.last_bind) if self.last_bind is not None else None,
                    gate_thresholds=dict(offset_soft_limit=self.offset_soft_limit,
                                         center_soft_limit=self.center_soft_limit,
                                         velocity_soft_limit=self.velocity_soft_limit,
                                         base_velocity_sigma_px_s=self.base_velocity_sigma_px_s,
                                         calibrated=False),
                    template_ready=bool(self.template_bank is not None and self.template_bank.ready),
                    last_diagnostics=dict(self.last_diagnostics))
