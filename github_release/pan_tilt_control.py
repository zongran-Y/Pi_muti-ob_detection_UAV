"""PID visual pan/tilt control; hardware imports are deferred until opened."""

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ServoConfig:
    pan_channel: int = 1
    tilt_channel: int = 0
    pan_home: float = 0.0
    tilt_home: float = 0.0
    pan_min: float = 0.0
    pan_max: float = 270.0
    tilt_min: float = 0.0
    tilt_max: float = 270.0
    pan_direction: int = 1
    tilt_direction: int = -1
    pan_gain: float = 40.0
    tilt_gain: float = 40.0
    max_speed: float = 30.0
    max_step: float = 5.0
    deadzone: float = 0.05
    smoothing: float = 0.5
    target_hold: float = 1.0
    target_gate: float = 0.35
    pulse_min: object = 500
    pulse_max: object = 2500
    actuation_range: float = 270.0
    pan_ki: float = 2.0
    tilt_ki: float = 2.0
    pan_kd: float = 1.0
    tilt_kd: float = 1.0
    pid_integral_limit: float = 0.5
    pid_derivative_smoothing: float = 0.25
    pid_reset_gap: float = 0.5
    velocity_feedforward: float = 0.0
    horizontal_fov: float = 62.0
    velocity_smoothing: float = 0.3
    feedforward_max_speed: float = 15.0
    feedforward_deadband: float = 1.0

    def __post_init__(self):
        values = (
            self.pan_home, self.tilt_home, self.pan_min, self.pan_max,
            self.tilt_min, self.tilt_max, self.pan_gain, self.tilt_gain,
            self.max_speed, self.max_step, self.deadzone, self.smoothing,
            self.target_hold, self.target_gate, self.actuation_range,
            self.pan_ki, self.tilt_ki, self.pan_kd, self.tilt_kd,
            self.pid_integral_limit, self.pid_derivative_smoothing, self.pid_reset_gap,
            self.velocity_feedforward, self.horizontal_fov, self.velocity_smoothing,
            self.feedforward_max_speed, self.feedforward_deadband,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Servo settings must be finite")
        if self.actuation_range <= 0:
            raise ValueError("Servo actuation range must be positive")
        if self.pan_channel == self.tilt_channel or not all(
            isinstance(channel, int) and 0 <= channel < 16
            for channel in (self.pan_channel, self.tilt_channel)
        ):
            raise ValueError("Pan and tilt need distinct channels between 0 and 15")
        for low, home, high in (
            (self.pan_min, self.pan_home, self.pan_max),
            (self.tilt_min, self.tilt_home, self.tilt_max),
        ):
            if not 0 <= low <= home <= high <= self.actuation_range or low == high:
                raise ValueError("Require 0 <= axis_min <= axis_home <= axis_max <= servo_range")
        if self.pan_direction not in (-1, 1) or self.tilt_direction not in (-1, 1):
            raise ValueError("Axis directions must be -1 or +1")
        if self.pan_gain <= 0 or self.tilt_gain <= 0 or self.max_speed <= 0 or self.max_step <= 0:
            raise ValueError("Gains and movement limits must be positive")
        if min(self.pan_ki, self.tilt_ki, self.pan_kd, self.tilt_kd, self.pid_integral_limit) < 0:
            raise ValueError("PID Ki, Kd and integral limit must be nonnegative")
        if not 0 < self.pid_derivative_smoothing <= 1 or self.pid_reset_gap <= 0:
            raise ValueError("PID derivative smoothing must be in (0, 1] and reset gap must be positive")
        if self.velocity_feedforward < 0 or not 0 < self.horizontal_fov < 180:
            raise ValueError("Feedforward gain must be nonnegative and horizontal FOV must be in (0, 180)")
        if not 0 < self.velocity_smoothing <= 1 or self.feedforward_max_speed <= 0 or self.feedforward_deadband < 0:
            raise ValueError("Invalid velocity smoothing, feedforward speed limit or deadband")
        if not 0 <= self.deadzone < 1 or not 0 < self.smoothing <= 1:
            raise ValueError("Invalid deadzone or smoothing coefficient")
        if self.target_hold <= 0 or self.target_gate <= 0:
            raise ValueError("Target hold time and matching gate must be positive")
        if (self.pulse_min is None) != (self.pulse_max is None):
            raise ValueError("Set both pulse-min and pulse-max, or neither")
        if self.pulse_min is not None and not 0 < self.pulse_min < self.pulse_max:
            raise ValueError("Require 0 < pulse-min < pulse-max")


class _TargetVelocity:
    """Local angular-rate proxy; command pose is NOT measured camera pose."""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.previous = None
        self.samples = 0
        self.pan_rate = self.tilt_rate = 0.0

    @property
    def ready(self):
        return self.samples >= 3

    def observe(self, ex, ey, width, height, pose, timestamp):
        tangent = math.tan(math.radians(self.config.horizontal_fov / 2))
                                                                                  
        bx = math.degrees(math.atan(ex * tangent))
        by = math.degrees(math.atan(ey * tangent * height / width))
        world_x = bx + self.config.pan_direction * pose[0]
        world_y = by + self.config.tilt_direction * pose[1]
        geometry = (width, height)
        previous = self.previous
        if previous is not None:
            gap = timestamp - previous[0]
            if gap <= 0 or gap > self.config.pid_reset_gap or geometry != previous[3]:
                self.reset()
                if gap <= 0:
                    return
                previous = None
        self.samples += 1
        if previous is not None:
            gap = timestamp - previous[0]
            vx, vy = (world_x - previous[1]) / gap, (world_y - previous[2]) / gap
            alpha = 1.0 if self.samples == 2 else self.config.velocity_smoothing
            self.pan_rate = alpha * vx + (1 - alpha) * self.pan_rate
            self.tilt_rate = alpha * vy + (1 - alpha) * self.tilt_rate
        self.previous = (timestamp, world_x, world_y, geometry)


class _AxisPID:
    """PID rate command with conditional integration for all actuator limits."""

    def __init__(self, kp, ki, kd, config):
        self.kp, self.ki, self.kd, self.config = kp, ki, kd, config
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_raw = None
        self.derivative = 0.0
        self.terms = dict(pid_p=0.0, pid_i=0.0, pid_d=0.0, pid_output_deg_s=0.0,
                          rate_deg_s=0.0, pid_saturated=False, feedforward_deg_s=0.0,
                          command_output_deg_s=0.0)

    def clear_derivative(self):
        """Clear D/error history, preserve bounded I state, and mark no new command."""
        bound = self.config.pid_integral_limit
        self.integral = max(-bound, min(bound, self.integral))
        self.previous_raw = None
        self.derivative = 0.0
        self.terms = dict(pid_p=0.0, pid_i=self.ki * self.integral, pid_d=0.0,
                          pid_output_deg_s=0.0, rate_deg_s=0.0,
                          pid_saturated=False, feedforward_deg_s=0.0,
                          command_output_deg_s=0.0)

    def step(self, angle, error, raw_error, direction, low, high, dt, move_dt, integrate=True, feedforward=0.0):
        inside = abs(raw_error) <= self.config.deadzone
        if inside:
            self.reset()
            if not feedforward:
                return angle, 0.0, False, "deadzone"
            error = raw_error = 0.0
        if self.previous_raw is not None and raw_error * self.previous_raw < 0:
            self.integral = 0.0
        raw_derivative = 0.0 if self.previous_raw is None else (raw_error - self.previous_raw) / dt
        self.previous_raw = None if inside else raw_error
        alpha = self.config.pid_derivative_smoothing
        self.derivative = alpha * raw_derivative + (1 - alpha) * self.derivative
        candidate = self.integral + error * dt if self.ki and integrate and not inside else self.integral
        bound = self.config.pid_integral_limit
        candidate = max(-bound, min(bound, candidate)) if self.ki else 0.0
        p_term, d_term = self.kp * error, self.kd * self.derivative

        def constrained(output):
            speed = max(-self.config.max_speed, min(self.config.max_speed, output))
            step = max(-self.config.max_step, min(self.config.max_step, direction * speed * move_dt))
            requested = angle + step
            updated = max(low, min(high, requested))
                                                                                     
            achieved = direction * (updated - angle) / dt
            return updated, updated != requested, achieved

        output = p_term + self.ki * candidate + d_term + feedforward
        updated, limited, achieved = constrained(output)
                                                                              
        saturated = abs(output - achieved) > 1e-7
        if saturated and error * (output - achieved) > 0:
            candidate = self.integral
            output = p_term + self.ki * candidate + d_term + feedforward
            updated, limited, achieved = constrained(output)
        self.integral = candidate
        self.terms = dict(pid_p=p_term, pid_i=self.ki * self.integral, pid_d=d_term,
                          pid_output_deg_s=output - feedforward, rate_deg_s=(updated - angle) / dt,
                          feedforward_deg_s=feedforward, command_output_deg_s=output,
                          pid_saturated=saturated or abs(output - achieved) > 1e-7)
        reason = "limit" if limited else ("moving" if abs(updated - angle) > 1e-6 else "small_step")
        if inside and reason == "moving":
            reason = "feedforward"
        return updated, updated - angle, limited, reason


class PanTiltController:
    def __init__(self, config, write_angle, clock=None):
        self.config = config
        self.write_angle = write_angle
                                                                                 
        self._clock = clock
        self.pan = config.pan_home
        self.tilt = config.tilt_home
        self.last_time = None
        self.target = None
        self.last_seen = None
        self.filtered_x = self.filtered_y = None
        self.pan_pid = _AxisPID(config.pan_gain, config.pan_ki, config.pan_kd, config)
        self.tilt_pid = _AxisPID(config.tilt_gain, config.tilt_ki, config.tilt_kd, config)
        self.motion = _TargetVelocity(config)
        self._pose_history = deque(maxlen=128)
        self._prediction_active = False
        self._manual_recovery_track_id = None

    def _reset_pid(self):
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.motion.reset()
        self.filtered_x = self.filtered_y = None
        self._prediction_active = False
        self._manual_recovery_track_id = None

    def reset_target_lock(self):
        """Clear target PID/EMA/velocity history, preserving commanded angles and pose history."""
        self.target = None
        self.last_seen = None
        self._reset_pid()

    def initialize(self, timestamp):
        self.pan, self.tilt = self.config.pan_home, self.config.tilt_home
        self.write_angle(self.config.pan_channel, self.pan)
        self.write_angle(self.config.tilt_channel, self.tilt)
        self.last_time = timestamp
        self.target = self.last_seen = None
        self._reset_pid()
        self._pose_history.clear()
        self._record_pose(self._clock() if self._clock is not None else timestamp)
        return self._state("home_commanded")

    def _record_pose(self, timestamp):
        self._pose_history.append((timestamp, self.pan, self.tilt))

    def _issue_angle(self, channel, angle, timestamp):
        issued_at = self._clock() if self._clock is not None else timestamp
        self.write_angle(channel, angle)
        return issued_at

    def _pose_at(self, timestamp):
        for issued_at, pan, tilt in reversed(self._pose_history):
            if issued_at <= timestamp:
                return pan, tilt
        return None

    def _state(self, status, target=None, pan_step=0.0, tilt_step=0.0,
               pan_limited=False, tilt_limited=False, pan_reason=None, tilt_reason=None,
               error_x=None, error_y=None):
        state = {
            "control_status": status,
            "control_track_id": target.track_id if target is not None else None,
            "control_position_valid": target is not None,
            "control_bbox": list(target.bbox) if target is not None else None,
            "control_confidence": (target.confidence if target is not None
                                   and getattr(target, "position_source", "detector") == "detector" else None),
            "control_position_source": getattr(target, "position_source", "detector") if target is not None else "none",
            "control_error_x": error_x, "control_error_y": error_y,
            "pan_angle_deg": self.pan, "tilt_angle_deg": self.tilt,
            "pan_step_deg": pan_step, "tilt_step_deg": tilt_step,
            "pan_limited": pan_limited, "tilt_limited": tilt_limited,
            "filtered_error_x": self.filtered_x, "filtered_error_y": self.filtered_y,
            "pan_reason": pan_reason or status, "tilt_reason": tilt_reason or status,
            "pan_at_limit": min(abs(self.pan - self.config.pan_min), abs(self.pan - self.config.pan_max)) < 1e-6,
            "tilt_at_limit": min(abs(self.tilt - self.config.tilt_min), abs(self.tilt - self.config.tilt_max)) < 1e-6,
            "target_pan_velocity_deg_s": self.motion.pan_rate,
            "target_tilt_velocity_deg_s": self.motion.tilt_rate,
            "velocity_ready": self.motion.ready, "velocity_samples": self.motion.samples,
        }
        state.update({"pan_" + key: value for key, value in self.pan_pid.terms.items()})
        state.update({"tilt_" + key: value for key, value in self.tilt_pid.terms.items()})
        return state

    def command_axis(self, axis, angle):
        """Small manual axis probes independent of detection and target selection."""
        if self.last_time is None:
            raise RuntimeError("Initialize servo home before an axis test")
        if axis not in ("pan", "tilt") or not math.isfinite(angle):
            raise ValueError("Specify pan/tilt and a finite angle")
        self._reset_pid()
        if axis == "pan":
            updated = max(self.config.pan_min, min(self.config.pan_max, angle))
            step = updated - self.pan
            issued_at = self._issue_angle(self.config.pan_channel, updated, self.last_time)
            self.pan = updated
            self._record_pose(issued_at)
            return self._state("manual_pan", pan_step=step, pan_limited=updated != angle,
                               pan_reason="limit" if updated != angle else "manual", tilt_reason="hold")
        updated = max(self.config.tilt_min, min(self.config.tilt_max, angle))
        step = updated - self.tilt
        issued_at = self._issue_angle(self.config.tilt_channel, updated, self.last_time)
        self.tilt = updated
        self._record_pose(issued_at)
        return self._state("manual_tilt", tilt_step=step, tilt_limited=updated != angle,
                           pan_reason="hold", tilt_reason="limit" if updated != angle else "manual")

    def _choose_target(self, candidates, width, height, timestamp,
                       preserve_target_identity=False):
        same_locked_id_present = (
            preserve_target_identity and self.target is not None
            and any(candidate.track_id == self.target.track_id for candidate in candidates))
        if (self.last_seen is not None
                and timestamp - self.last_seen > self.config.target_hold
                and not same_locked_id_present):
            self.target = None
            self._reset_pid()
        if self.target is None:
            return candidates[0] if len(candidates) == 1 else None
        previous = self.target
        px = (previous.bbox[0] + previous.bbox[2]) / 2
        py = (previous.bbox[1] + previous.bbox[3]) / 2
        identities, spatial = [], []
        for candidate in candidates:
            if candidate.class_id != previous.class_id:
                continue
            cx = (candidate.bbox[0] + candidate.bbox[2]) / 2
            cy = (candidate.bbox[1] + candidate.bbox[3]) / 2
            distance = math.hypot((cx - px) / (width / 2), (cy - py) / (height / 2))
            same_id = candidate.track_id is not None and candidate.track_id == previous.track_id
            if same_id:
                identities.append(candidate)
            if distance <= self.config.target_gate:
                spatial.append(candidate)
        if identities:
            return identities[0] if len(identities) == 1 else None
        return spatial[0] if len(spatial) == 1 else None

    def update(self, candidates, width, height, timestamp, observed_at=None,
               hold_reason=None, preserve_target_identity=False, *, selected_track_id=None):
        if self.last_time is None:
            raise RuntimeError("Initialize servo home before tracking")
        if width <= 0 or height <= 0 or not math.isfinite(timestamp):
            raise ValueError("Invalid image dimensions or timestamp")
        observed_at = timestamp if observed_at is None else observed_at
        if not math.isfinite(observed_at) or observed_at > timestamp:
            raise ValueError("Observation time must be finite and no later than control time")

        candidate = candidates[0] if len(candidates) == 1 else None
        support_at = getattr(candidate, "last_visual_support_at", None)
        valid_until = getattr(candidate, "valid_until", None)
        support_at = support_at if support_at is not None and math.isfinite(support_at) else None
        valid_until = valid_until if valid_until is not None and math.isfinite(valid_until) else None

        def result(status, *args, reject_reason="", **kwargs):
            state = self._state(status, *args, **kwargs)
            state.update(control_timestamp=timestamp, control_selected_track_id=selected_track_id,
                         control_bound_track_id=self.target.track_id if self.target is not None else None,
                         control_visual_support_at=support_at,
                         control_visual_support_age_s=(timestamp - support_at if support_at is not None else None),
                         control_valid_until=valid_until, control_reject_reason=reject_reason)
            return state

        dt = timestamp - self.last_time
        hold_status = {"ambiguous": "ambiguous_hold",
                       "position_invalid": "position_invalid_hold"}.get(hold_reason)
        if not preserve_target_identity and hold_reason != "position_invalid":
            self._manual_recovery_track_id = None
        if (hold_reason == "position_invalid" and preserve_target_identity
                and self.target is not None):
            self._manual_recovery_track_id = self.target.track_id
        if dt <= 0:
            if hold_status is not None:
                self.pan_pid.clear_derivative()
                self.tilt_pid.clear_derivative()
                return result(hold_status, reject_reason=hold_reason)
            self._reset_pid()
            return result("timing_hold", reject_reason="nonpositive_control_dt")
        self.last_time = timestamp
        preserve_pid_hold = hold_status is not None
        stale = dt > self.config.pid_reset_gap
        if stale and not preserve_pid_hold:
            self._reset_pid()
        if preserve_pid_hold:
                                                                              
                                                                             
                                                                        
            self.pan_pid.clear_derivative()
            self.tilt_pid.clear_derivative()
            return result(hold_status, reject_reason=hold_reason)
        if (candidate is not None
                and getattr(candidate, "semantic_state", "verified") == "invalid"):
                                                                            
                                                                         
            self.pan_pid.clear_derivative()
            self.tilt_pid.clear_derivative()
            return result("semantic_invalid_hold", reject_reason="semantic_invalid")
                                                                          
        move_dt = min(0.2, dt)
        if selected_track_id is not None and candidates and (
                candidate is None or candidate.track_id != selected_track_id
                or (self.target is not None and self.target.track_id != selected_track_id)):
            self._reset_pid()
            return result("lost_hold", reject_reason="selected_id_mismatch")
        fresh_candidates = [candidate for candidate in candidates
                            if (selected_track_id is None and getattr(candidate, 'valid_until', None) is None)
                            or (getattr(candidate, 'valid_until', None) is not None
                                and math.isfinite(candidate.valid_until) and timestamp <= candidate.valid_until)]
        if candidates and not fresh_candidates:
            self._reset_pid()
            return result('stale_position_hold', reject_reason=(
                "invalid_position_deadline" if candidate is not None and valid_until is None
                else "position_expired"))
        candidates = fresh_candidates
        preserving_recovery = bool(
            preserve_target_identity and self._manual_recovery_track_id is not None)
        chosen = self._choose_target(candidates, width, height, timestamp,
                                     preserve_target_identity=preserving_recovery)
        if chosen is None:
            status = ("ambiguous_hold" if hold_reason == "ambiguous" or len(candidates) > 1
                      else "lost_hold")
            self._reset_pid()
            return result(status, reject_reason=("no_candidate" if not candidates else "target_not_unique_or_matched"))
        predicted = getattr(chosen, "position_source", "detector") == "prediction"
        detector_at = getattr(chosen, "last_detection_at", None)
        support_at = getattr(chosen, "last_visual_support_at", None)
        lk_supported = bool(predicted and detector_at is not None and support_at is not None
                            and support_at > detector_at + 1e-6
                            and 0 <= timestamp - support_at <= .12)
        if (predicted and (self.target is None or chosen.track_id != self.target.track_id)
                and selected_track_id is None):
            self._reset_pid()
            return result("lost_hold", reject_reason="prediction_without_selected_id")
        same_locked_identity = (preserving_recovery
                                and chosen.track_id == self._manual_recovery_track_id)
        if (((predicted and not lk_supported)
             or (self._prediction_active and not predicted)
             or getattr(chosen, 'motion_reset', False))
                and not same_locked_identity):
            self._reset_pid()
        self._prediction_active = predicted
        self.target = chosen
        if not predicted:
            self.last_seen = timestamp
        cx, cy = (chosen.bbox[0] + chosen.bbox[2]) / 2, (chosen.bbox[1] + chosen.bbox[3]) / 2
        ex, ey = (cx - width / 2) / (width / 2), (cy - height / 2) / (height / 2)
        if self.config.velocity_feedforward and not predicted:
            pose = self._pose_at(observed_at)
            if pose is None or timestamp - observed_at > self.config.pid_reset_gap:
                self.motion.reset()
            else:
                self.motion.observe(ex, ey, width, height, pose, observed_at)

        def feedforward(rate):
            if not self.config.velocity_feedforward or not self.motion.ready:
                return 0.0
            output = rate * self.config.velocity_feedforward
            bound = self.config.feedforward_max_speed
            output = max(-bound, min(bound, output))
            return output if abs(output) >= self.config.feedforward_deadband else 0.0

        pan_ff, tilt_ff = feedforward(self.motion.pan_rate), feedforward(self.motion.tilt_rate)
        alpha = self.config.smoothing
        self.filtered_x = ex if self.filtered_x is None else alpha * ex + (1 - alpha) * self.filtered_x
        self.filtered_y = ey if self.filtered_y is None else alpha * ey + (1 - alpha) * self.filtered_y
                                                                             
        if abs(ex) <= self.config.deadzone:
            self.filtered_x = 0.0
        elif ex * self.filtered_x < 0:
            self.filtered_x = ex
        if abs(ey) <= self.config.deadzone:
            self.filtered_y = 0.0
        elif ey * self.filtered_y < 0:
            self.filtered_y = ey
        new_pan, pan_step, pan_limited, pan_reason = self.pan_pid.step(
            self.pan, self.filtered_x * (0.5 if predicted and not lk_supported else 1.0), ex, self.config.pan_direction,
            self.config.pan_min, self.config.pan_max, dt, move_dt, integrate=not stale and not predicted,
            feedforward=pan_ff,
        )
        new_tilt, tilt_step, tilt_limited, tilt_reason = self.tilt_pid.step(
            self.tilt, self.filtered_y * (0.5 if predicted and not lk_supported else 1.0), ey, self.config.tilt_direction,
            self.config.tilt_min, self.config.tilt_max, dt, move_dt, integrate=not stale and not predicted,
            feedforward=tilt_ff,
        )
        if abs(pan_step) > 1e-6:
            issued_at = self._issue_angle(self.config.pan_channel, new_pan, timestamp)
            self.pan = new_pan
            self._record_pose(issued_at)
        if abs(tilt_step) > 1e-6:
            issued_at = self._issue_angle(self.config.tilt_channel, new_tilt, timestamp)
            self.tilt = new_tilt
            self._record_pose(issued_at)
        centered = abs(ex) <= self.config.deadzone and abs(ey) <= self.config.deadzone
        status = "angle_limit" if pan_limited or tilt_limited else ("centered" if centered else "tracking")
        if centered and not (pan_limited or tilt_limited) and (abs(pan_step) > 1e-6 or abs(tilt_step) > 1e-6):
            status = "centered_tracking"
        if predicted:
            status = "prediction_tracking" if abs(pan_step) + abs(tilt_step) > 1e-6 else "prediction_hold"
        return result(status, chosen, pan_step, tilt_step, pan_limited, tilt_limited,
                      pan_reason, tilt_reason, error_x=ex, error_y=ey)


def open_servo_controller(config):
    from adafruit_servokit import ServoKit

    kit = ServoKit(channels=16)
    for channel in (config.pan_channel, config.tilt_channel):
        kit.servo[channel].actuation_range = config.actuation_range
        if config.pulse_min is not None:
            kit.servo[channel].set_pulse_width_range(config.pulse_min, config.pulse_max)

    def write_angle(channel, angle):
        kit.servo[channel].angle = angle

    return PanTiltController(config, write_angle, clock=time.perf_counter)
