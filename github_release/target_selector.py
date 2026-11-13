"""Fail-safe selector for one camera-control target over MOT track views."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TargetSelection:
    active_track_id: object
    target: object
    status: str
    active_epoch: int
    pending_track_id: object = None
    switched: bool = False
    release_reason: str = ""
    ambiguous_count: int = 0
    ambiguous_duration_s: float = 0.0
    manual_request_result: str = ""
    manual_active_lock: bool = False
    candidate_diagnostics: tuple = ()
    pending_confirmations: int = 0


class TargetSelector:
    """Retain one active ID and require independent evidence before switching."""

    def __init__(self, control_horizon=.35, active_hold_seconds=.35,
                 switch_confidence=.15, switch_confirmations=2,
                 switch_window_seconds=1.25, max_position_sigma_fraction=2.0):
        values = (control_horizon, active_hold_seconds, switch_confidence,
                  switch_window_seconds, max_position_sigma_fraction)
        if (not all(math.isfinite(value) for value in values)
                or not 0 < control_horizon <= .35 or active_hold_seconds < control_horizon
                or not 0 <= switch_confidence <= 1 or switch_confirmations < 2
                or switch_window_seconds <= 0 or max_position_sigma_fraction <= 0):
            raise ValueError("Invalid active target selection configuration")
        self.control_horizon = float(control_horizon)
        self.active_hold_seconds = float(active_hold_seconds)
        self.switch_confidence = float(switch_confidence)
        self.switch_confirmations = int(switch_confirmations)
        self.switch_window_seconds = float(switch_window_seconds)
        self.max_position_sigma_fraction = float(max_position_sigma_fraction)
        self.reset()

    def reset(self):
        self.active_track_id = None
        self.active_epoch = 0
        self._ever_acquired = False
        self._released_at = None
        self._last_class_id = None
        self._clear_pending()
        self._requested_id = None
        self._request_pending = False
        self._manual_active_lock = False
        self._manual_lock_expired = False
        self.last_manual_request_result = ""
        self._ambiguous_since = None
        self._ambiguous_duration_s = 0.0
        self._current_candidate_diagnostics = ()
        self._last_candidate_diagnostics = ()
        self.last_status = "searching"
        self.last_release_reason = ""

    def request_acquire(self, track_id=None):
        """Queue a confirmed/lost ID request; repeated calls before select() are last-wins."""
        self._requested_id = track_id
        self._request_pending = True
        self.last_manual_request_result = "pending"

    def _selection(self, timestamp, status, target=None, pending_track_id=None,
                   switched=False, release_reason="", ambiguous_count=0,
                   manual_request_result=""):
        if status == "ambiguous":
            if self._ambiguous_since is None:
                self._ambiguous_since = float(timestamp)
            self._ambiguous_duration_s = max(0.0, float(timestamp) - self._ambiguous_since)
        else:
            self._ambiguous_since = None
            self._ambiguous_duration_s = 0.0
        self.last_status = status
        self._last_candidate_diagnostics = tuple(self._current_candidate_diagnostics)
        return TargetSelection(self.active_track_id, target, status, self.active_epoch,
                               pending_track_id=pending_track_id,
                               switched=switched, release_reason=release_reason,
                               ambiguous_count=ambiguous_count,
                               ambiguous_duration_s=self._ambiguous_duration_s,
                               manual_request_result=manual_request_result,
                               manual_active_lock=self._manual_active_lock,
                               candidate_diagnostics=self._last_candidate_diagnostics,
                               pending_confirmations=len(self._pending_observations))

    def _set_active(self, track, manual_lock=False):
        changed = self.active_track_id != track.track_id
        self.active_track_id = track.track_id
        self._last_class_id = track.class_id
        self._ever_acquired = True
        self._released_at = None
        self._manual_active_lock = bool(manual_lock)
        self._manual_lock_expired = False
        self._clear_pending()
        if changed:
            self.active_epoch += 1
        return changed

    def _clear_pending(self):
        self._pending_id = None
        self._pending_observations = []

    def _expire_pending(self, by_id, timestamp):
        if self._pending_id is None:
            return
        track = by_id.get(self._pending_id)
        if (track is None or track.status not in ("confirmed", "lost")
                or track.class_id != self._last_class_id or not self._identity_confirmed(track)):
            self._clear_pending()
            return
        self._pending_observations = [
            record for record in self._pending_observations
            if timestamp - record[1] <= self.switch_window_seconds]
        if not self._pending_observations:
            self._clear_pending()

    def _release(self, timestamp, reason):
        if self.active_track_id is not None:
            self.active_epoch += 1
        if self._manual_active_lock:
            self._manual_lock_expired = True
        self.active_track_id = None
        self._released_at = float(timestamp)
        self.last_release_reason = reason
        self._manual_active_lock = False
        self._clear_pending()

    @staticmethod
    def _identity_confirmed(track):
        flag = getattr(track, "identity_confirmed", None)
        if flag is not None:
            return bool(flag)
        if getattr(track, "status", None) == "confirmed":
            return True
        try:
            if int(getattr(track, "hits", 0)) >= 2:
                return True
        except (TypeError, ValueError):
            pass
        records = getattr(track, "recent_detection_records", ())
        try:
            return len({int(record[0]) for record in records}) >= 2
        except (TypeError, ValueError, IndexError):
            return False

    def _control_validity_reason(self, track, timestamp):
        if track is None or track.status not in ("confirmed", "lost"):
            return "lifecycle_ineligible"
        if getattr(track, "semantic_state", "verified") == "invalid":
            return "semantic_invalid"
        support_at = getattr(track, "last_visual_support_at", track.last_detection_at)
        age = max(0.0, timestamp - support_at)
        if age > self.control_horizon:
            return "position_stale"
        valid_until = getattr(track, "valid_until", None)
        if valid_until is not None and timestamp > valid_until:
            return "detector_anchor_stale"
        try:
            variance_x, variance_y = float(track.covariance[0]), float(track.covariance[1])
            width = float(track.bbox[2] - track.bbox[0])
            height = float(track.bbox[3] - track.bbox[1])
        except (TypeError, ValueError, IndexError):
            return "invalid_state"
        if not all(math.isfinite(value) for value in (variance_x, variance_y, width, height)):
            return "invalid_state"
        if min(width, height) <= 0 or min(variance_x, variance_y) < 0:
            return "invalid_state"
        max_sigma = max(6.0, min(width, height) * self.max_position_sigma_fraction)
        if math.sqrt(max(variance_x, variance_y)) > max_sigma:
            return "position_uncertain"
        return "eligible"

    def _valid_control_track(self, track, timestamp):
        return self._control_validity_reason(track, timestamp) == "eligible"

    def _candidate_diagnostics(self, views, timestamp):
        diagnostics = []
        for view in views:
            identity_confirmed = self._identity_confirmed(view)
            reason = (self._control_validity_reason(view, timestamp)
                      if identity_confirmed else "identity_unconfirmed")
            diagnostics.append(dict(
                track_id=str(view.track_id), status=str(view.status),
                hits=int(getattr(view, "hits", 0)),
                identity_confirmed=identity_confirmed,
                detection_age_s=float(getattr(view, "age_s", 0.0)),
                semantic_age_s=float(getattr(view, "semantic_age_s", 0.0)),
                semantic_state=str(getattr(view, "semantic_state", "verified")),
                full_frame_unmatched_count=int(getattr(
                    view, "full_frame_unmatched_count", 0)),
                visual_support_age_s=float(getattr(view, "visual_support_age_s", 0.0)),
                eligible=reason == "eligible", reason=reason))
        return tuple(diagnostics)

    def _switch_candidates(self, eligible_candidates):
        return [view for view in eligible_candidates
                if view.status in ("confirmed", "lost")
                and self._identity_confirmed(view)
                and getattr(view, "semantic_state", "verified") == "verified"
                and view.track_id != self.active_track_id
                and view.class_id == self._last_class_id
                and view.confidence >= self.switch_confidence]

    @staticmethod
    def _candidate_rank(track):
        """Prefer the earliest consistently observed ID, then stronger/cleaner state."""
        records = tuple(getattr(track, "recent_detection_records", ()) or ())
        first_sequence = getattr(track, "first_detection_sequence", None)
        if first_sequence is None:
            first_sequence = min((int(record[0]) for record in records), default=2 ** 63 - 1)
        first_sequence = int(first_sequence)
        mean_confidence = (sum(float(record[1]) for record in records) / len(records)
                           if records else float(getattr(track, "confidence", 0.0)))
        try:
            sigma = max(math.sqrt(max(0.0, float(track.covariance[0]))),
                        math.sqrt(max(0.0, float(track.covariance[1]))))
        except (TypeError, ValueError, IndexError):
            sigma = math.inf
        return first_sequence, -mean_confidence, sigma, str(track.track_id)

    def _consider_switch(self, candidates, timestamp):
        if not candidates:
                                                                                            
            self._clear_pending()
            return None, "searching"
                                                                         
                                                                             
                                                                             
                                                               
        candidate = min(candidates, key=self._candidate_rank)
        self._clear_pending()
        self._set_active(candidate)
        return candidate, "switched"

    def select(self, track_views, timestamp, active_track_retained=None):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Selector timestamp must be finite and nonnegative")
        views = list(track_views or ())
        self._current_candidate_diagnostics = self._candidate_diagnostics(views, timestamp)
        eligible_ids = {item["track_id"] for item in self._current_candidate_diagnostics
                        if item["eligible"]}
        eligible_for_acquire = [view for view in views
                                if str(view.track_id) in eligible_ids]
        by_id = {view.track_id: view for view in views}
        if active_track_retained is None:
            active_track_retained = self.active_track_id in by_id
        else:
            active_track_retained = bool(active_track_retained)
        switched = False
        release_reason = ""
        manual_request_result = ""
        active = by_id.get(self.active_track_id)

        if self._request_pending:
            requested_id = self._requested_id
            self._request_pending = False
            self._requested_id = None
            requested = (by_id.get(str(requested_id)) if requested_id is not None
                         else eligible_for_acquire[0]
                         if len(eligible_for_acquire) == 1 else None)
            if (requested_id is not None
                    and str(requested_id) == str(self.active_track_id)):
                manual_request_result = "ignored_same_active"
            elif (requested is not None
                  and requested.status in ("confirmed", "lost")
                  and self._identity_confirmed(requested)):
                switched = self._set_active(requested, manual_lock=requested_id is not None)
                manual_request_result = "accepted"
                self.last_manual_request_result = manual_request_result
                target = requested if self._valid_control_track(requested, timestamp) else None
                status = "manual_active" if switched and target is not None else (
                    "active" if target is not None else "lost_hold")
                return self._selection(timestamp, status, target=target,
                                       switched=switched,
                                       ambiguous_count=0,
                                       manual_request_result=manual_request_result)
            elif requested_id is None:
                manual_request_result = "rejected_not_unique"
            else:
                manual_request_result = "rejected_not_confirmed"
            self.last_manual_request_result = manual_request_result
            if manual_request_result.startswith("rejected") and self.active_track_id is None:
                status = "ambiguous" if len(eligible_for_acquire) > 1 else "searching"
                return self._selection(
                    timestamp, status,
                    ambiguous_count=(len(eligible_for_acquire)
                                     if status == "ambiguous" else 0),
                    manual_request_result=manual_request_result)

        if self.active_track_id is not None:
            if active is None or active.status == "deleted":
                if active is None and active_track_retained and not any(
                        self._switch_candidates(eligible_for_acquire)):
                                                                         
                                                                      
                    return self._selection(timestamp, "lost_hold",
                                           manual_request_result=manual_request_result)
                release_reason = ("manual_active_track_deleted" if self._manual_active_lock
                                  else "active_track_deleted")
                self._release(timestamp, release_reason)
                active = None
            elif self._valid_control_track(active, timestamp):
                                                                                
                                                                            
                                                                                 
                return self._selection(timestamp, "active", target=active,
                                       manual_request_result=manual_request_result)
            else:
                invalid_reason = self._control_validity_reason(active, timestamp)
                if self._manual_active_lock:
                    return self._selection(timestamp, "lost_hold",
                                           manual_request_result=manual_request_result)
                if timestamp - active.last_detection_at <= self.active_hold_seconds:
                    return self._selection(timestamp, "lost_hold",
                                           manual_request_result=manual_request_result)
                if active_track_retained:
                                                                            
                                                                                  
                    return self._selection(timestamp, "lost_hold",
                                           manual_request_result=manual_request_result)
                release_reason = "active_control_invalid:%s" % invalid_reason
                self._release(timestamp, release_reason)
                active = None

        if self._manual_lock_expired:
            status = "ambiguous" if len(eligible_for_acquire) > 1 else "searching"
            return self._selection(timestamp, status,
                                   release_reason=self.last_release_reason,
                                   ambiguous_count=(len(eligible_for_acquire)
                                                    if status == "ambiguous" else 0),
                                   manual_request_result=manual_request_result)
        eligible = self._switch_candidates(eligible_for_acquire)
        if self.last_status == "ambiguous" and len(eligible_for_acquire) == 1:
            chosen = min(eligible_for_acquire, key=self._candidate_rank)
            switched = self._set_active(chosen)
            target = chosen if self._valid_control_track(chosen, timestamp) else None
            status = "active" if target is not None else "lost_hold"
            return self._selection(timestamp, status, target=target, switched=switched,
                                   manual_request_result=manual_request_result)
        if self.last_status == "ambiguous" and len(eligible_for_acquire) > 1:
            return self._selection(timestamp, "ambiguous", target=None,
                                   ambiguous_count=len(eligible_for_acquire),
                                   manual_request_result=manual_request_result)
        if not self._ever_acquired and self._released_at is None:
            if eligible_for_acquire:
                chosen = min(eligible_for_acquire, key=self._candidate_rank)
                switched = self._set_active(chosen)
                target = chosen if self._valid_control_track(chosen, timestamp) else None
                status = "active" if target is not None else "lost_hold"
                return self._selection(timestamp, status, target=target,
                                   switched=switched,
                                   manual_request_result=manual_request_result)
            return self._selection(timestamp, "searching",
                                   manual_request_result=manual_request_result)

                                                                             
                                                                           
                                                             
        if self.active_track_id is None and self._released_at is not None:
            eligible_replacements = self._switch_candidates(eligible_for_acquire)
            if len(eligible_replacements) == 1:
                chosen = eligible_replacements[0]
                switched = self._set_active(chosen)
                target = chosen if self._valid_control_track(chosen, timestamp) else None
                return self._selection(timestamp,
                                       "active" if target is not None else "lost_hold",
                                       target=target, switched=switched,
                                       manual_request_result=manual_request_result)
            if len(eligible_replacements) > 1:
                return self._selection(timestamp, "ambiguous", target=None,
                                       ambiguous_count=len(eligible_replacements),
                                       manual_request_result=manual_request_result)
            return self._selection(timestamp, "searching",
                                   manual_request_result=manual_request_result)

        self._expire_pending(by_id, timestamp)
        target, status = self._consider_switch(eligible, timestamp)
        if target is not None:
            switched = True
            target_view = target if self._valid_control_track(target, timestamp) else None
            if target_view is None:
                status = "lost_hold"
        else:
            target_view = None
        return self._selection(timestamp, status, target=target_view,
                               pending_track_id=self._pending_id,
                               switched=switched, release_reason=release_reason,
                               ambiguous_count=len(eligible) if status == "ambiguous" else 0,
                               manual_request_result=manual_request_result)

    def summary(self):
        latest = self._pending_observations[-1] if self._pending_observations else (None, None)
        return dict(active_track_id=self.active_track_id,
                    active_epoch=self.active_epoch,
                    ever_acquired=self._ever_acquired,
                    released_at=self._released_at,
                    pending_track_id=self._pending_id,
                    manual_acquire_pending=self._request_pending,
                    last_manual_request_result=self.last_manual_request_result,
                    pending_confirmations=len(self._pending_observations),
                    pending_last_sequence=latest[0],
                    last_status=self.last_status,
                    last_release_reason=self.last_release_reason,
                    ambiguous_since=self._ambiguous_since,
                    ambiguous_duration_s=self._ambiguous_duration_s,
                    control_horizon_s=self.control_horizon,
                    active_hold_seconds=self.active_hold_seconds,
                    manual_active_lock=self._manual_active_lock,
                    manual_lock_expired=self._manual_lock_expired,
                    switch_confidence=self.switch_confidence,
                    switch_confirmations=self.switch_confirmations,
                    switch_confirmation_policy="identity_confirmation_only",
                    switch_window_seconds=self.switch_window_seconds,
                    last_candidate_diagnostics=list(self._last_candidate_diagnostics))
