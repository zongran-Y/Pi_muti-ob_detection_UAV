"""ROI pyramidal LK plus guarded RANSAC similarity motion (no deep model)."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlowConfig:
    center_only: bool = False
    roi_scale: float = 3.0
    min_roi_side: int = 160
    max_roi_side: int = 480
    max_points: int = 80
    min_inliers: int = 6
    min_inlier_ratio: float = .55
    fb_threshold: float = 1.5
    ransac_threshold: float = 2.5
    max_reprojection_error: float = 2.0
    min_scale: float = .75
    max_scale: float = 1.33
    max_rotation: float = 25.0

    def __post_init__(self):
        floats = (self.roi_scale, self.min_inlier_ratio, self.fb_threshold, self.ransac_threshold,
                  self.max_reprojection_error, self.min_scale, self.max_scale, self.max_rotation)
        if (not all(math.isfinite(v) for v in floats) or not 1.5 <= self.roi_scale <= 6
                or not 4 <= self.min_inliers <= self.max_points <= 200
                or not .1 <= self.min_inlier_ratio <= 1 or self.min_roi_side < 32
                or self.max_roi_side < self.min_roi_side or self.fb_threshold <= 0
                or self.ransac_threshold <= 0 or self.max_reprojection_error <= 0
                or not 0 < self.min_scale <= 1 <= self.max_scale or not 0 < self.max_rotation <= 90):
            raise ValueError('Invalid ROI flow configuration')


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    matrix = np.asarray(matrix, dtype=float)
    return points @ matrix[:, :2].T + matrix[:, 2]


def search_roi(bbox, shape, config):
    height, width = shape[:2]
    bw, bh = bbox[2]-bbox[0], bbox[3]-bbox[1]
    rw = min(width, max(bw+32, min(config.max_roi_side, max(config.min_roi_side, bw*config.roi_scale))))
    rh = min(height, max(bh+32, min(config.max_roi_side, max(config.min_roi_side, bh*config.roi_scale))))
    rw, rh = int(math.ceil(rw)), int(math.ceil(rh))
    x1 = max(0, min(width-rw, math.floor((bbox[0]+bbox[2]-rw)/2)))
    y1 = max(0, min(height-rh, math.floor((bbox[1]+bbox[3]-rh)/2)))
    return x1, y1, x1+rw, y1+rh


def assess_motion(matrix, old, new, mask, bbox, config):
    result = dict(ok=False, reason='affine_estimation_failed', inlier_count=0,
                  inlier_ratio=0., reprojection_error=None, scale=None, rotation_deg=None)
    if matrix is None or mask is None:
        return result
    matrix = np.asarray(matrix, dtype=float)
    old, new = np.asarray(old).reshape(-1, 2), np.asarray(new).reshape(-1, 2)
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if (matrix.shape != (2, 3) or len(old) != len(new) or len(mask) != len(old)
            or not np.isfinite(matrix).all() or not np.isfinite(old).all() or not np.isfinite(new).all()):
        result['reason'] = 'invalid_affine_geometry'
        return result
    count = int(mask.sum())
    result.update(inlier_count=count, inlier_ratio=count/max(1, len(old)))
    if count < config.min_inliers or result['inlier_ratio'] < config.min_inlier_ratio:
        result['reason'] = 'insufficient_affine_inliers'
        return result
    linear = matrix[:, :2]
    scale = float(np.linalg.norm(linear[:, 0]))
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    result.update(scale=scale, rotation_deg=rotation)
    if (np.linalg.det(linear) <= 0 or not config.min_scale <= scale <= config.max_scale
            or abs(rotation) > config.max_rotation
            or not np.allclose(linear.T @ linear, np.eye(2)*scale**2, atol=1e-3)):
        result['reason'] = 'affine_motion_gate'
        return result
    normalized = old[mask] / np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])
    span = np.ptp(normalized, axis=0)
    covariance = np.cov(normalized.T)
    coverage = math.sqrt(max(0., float(np.linalg.det(covariance))))
    if min(span) < .15 or coverage < .0025:
        result['reason'] = 'poor_spatial_distribution'
        return result
    residuals = np.linalg.norm(transform_points(old[mask], matrix)-new[mask], axis=1)
    error = float(np.median(residuals))
    result['reprojection_error'] = error
    if error > config.max_reprojection_error:
        result['reason'] = 'reprojection_error_gate'
        return result
    result.update(ok=True, reason='')
    return result


def inside_quad(points, quad, margin=3.):
    points = np.asarray(points).reshape(-1, 2)
    edges = np.roll(quad, -1, axis=0)-quad
    offsets = points[:, None, :]-quad[None, :, :]
    cross = edges[None, :, 0]*offsets[:, :, 1]-edges[None, :, 1]*offsets[:, :, 0]
    slack = margin*np.linalg.norm(edges, axis=1)
    return (cross >= -slack).all(axis=1) | (cross <= slack).all(axis=1)


class ROIAffineTracker:
    method = 'roi_lk_ransac'

    def __init__(self, cv2, config=None):
        self.cv2 = cv2
        self.config = config or FlowConfig()
        self.last_failure_reason = ''
        self.good_point_count = 0
        self.diagnostics = {}
        self._motion_hint = None

    def set_motion_hint(self,bbox):
        """One-frame hint from separately validated target appearance."""
        if (len(bbox)!=4 or not all(math.isfinite(v) for v in bbox)
                or bbox[2]<=bbox[0] or bbox[3]<=bbox[1]):
            raise ValueError('Invalid motion hint')
        self._motion_hint=tuple(float(v) for v in bbox)

    def reanchor(self,frame,bbox):
        """Keep current-frame points after a small confirmed box correction."""
        if (frame.shape[:2]!=self.shape or len(bbox)!=4
                or not all(math.isfinite(v) for v in bbox)):
            return False
        w,h=self.bbox[2]-self.bbox[0],self.bbox[3]-self.bbox[1]
        bw,bh=bbox[2]-bbox[0],bbox[3]-bbox[1]
        dx=((bbox[0]+bbox[2])-(self.bbox[0]+self.bbox[2]))/2
        dy=((bbox[1]+bbox[3])-(self.bbox[1]+self.bbox[3]))/2
        x,y,r,b=self.roi
        if (not .8<=bw/w<=1.25 or not .8<=bh/h<=1.25
                or math.hypot(dx,dy)>.15*min(w,h)
                or bbox[0]<x or bbox[1]<y or bbox[2]>r or bbox[3]>b):
            return False
        points=self.points.reshape(-1,2)+np.array(self.roi[:2])
        inside=((points[:,0]>=bbox[0])&(points[:,0]<=bbox[2])
                &(points[:,1]>=bbox[1])&(points[:,1]<=bbox[3]))
        if int(inside.sum())<self.config.min_inliers:
            return False
        self.points=self.points[inside].copy()
        self.bbox=tuple(float(v) for v in bbox)
        x1,y1,x2,y2=self.bbox
        self.quad=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=float)
        return True

    def _gray(self, frame, roi):
        x1, y1, x2, y2 = roi
        patch = frame[y1:y2, x1:x2]
        gray = patch if patch.ndim == 2 else self.cv2.cvtColor(patch, self.cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(gray)

    def _features(self, gray, roi, retained=None):
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        polygon = np.rint(self.quad-np.array(roi[:2])).astype(np.int32)
        self.cv2.fillConvexPoly(mask, polygon, 255)
        count = 0 if retained is None else len(retained)
        if count >= self.config.max_points:
            return None
        if retained is not None:
            for point in retained.reshape(-1, 2):
                self.cv2.circle(mask, tuple(round(float(v)) for v in point), 4, 0, -1)
        return self.cv2.goodFeaturesToTrack(gray, maxCorners=self.config.max_points-count,
                                           qualityLevel=.01, minDistance=3, mask=mask, blockSize=3)

    def init(self, frame, bbox):
        self._motion_hint = None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        if not all(math.isfinite(v) for v in bbox) or x2-x1 < 6 or y2-y1 < 6:
            self.last_failure_reason = 'invalid_initial_box'
            return False
        self.bbox = tuple(float(v) for v in bbox)
        self.quad = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=float)
        self.shape = height, width
        self.roi = search_roi(self.bbox, self.shape, self.config)
        self.gray = self._gray(frame, self.roi)
        self.points = self._features(self.gray, self.roi)
        self.frames = 0
        self.good_point_count = len(self.points) if self.points is not None else 0
        self.last_failure_reason = '' if self.good_point_count >= self.config.min_inliers else 'insufficient_features'
        self.diagnostics = dict(roi_area=self.gray.size, consistent_points=self.good_point_count)
        return not self.last_failure_reason

    def _failed(self, reason):
        self.last_failure_reason = reason
        self.diagnostics['failure_reason'] = reason
        return False, None

    def update(self, frame):
        hint,self._motion_hint=self._motion_hint,None
        self.good_point_count = 0
        self.diagnostics = dict(roi_area=self.gray.size, fwd_ok=0, bwd_ok=None, roundtrip_ok=None,
                                ransac_reject_reason='not_run')
        if frame.shape[:2] != self.shape:
            return self._failed('frame_shape_changed')
        cv2 = self.cv2
        gray = self._gray(frame, self.roi)
        settings = dict(winSize=(21,21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **settings)
        if moved is None or forward is None:
            return self._failed('forward_flow_failed')
        forward_points = np.asarray(moved).reshape(-1, 2)
        forward_status = np.asarray(forward).reshape(-1)
        if len(forward_points) == len(forward_status):
            fwd_ok = (forward_status != 0) & np.isfinite(forward_points).all(axis=1)
            self.diagnostics['fwd_ok'] = int(fwd_ok.sum())
        backward, status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, moved, None, **settings)
        if backward is None or status is None:
            return self._failed('backward_flow_failed')
        old = self.points.reshape(-1,2)
        new = np.asarray(moved).reshape(-1,2)
        back = np.asarray(backward).reshape(-1,2)
        if len(new) != len(old) or len(back) != len(old):
            return self._failed('flow_shape_mismatch')
        fwd_ok = ((np.asarray(forward).reshape(-1) != 0) & np.isfinite(new).all(axis=1))
        bwd_ok = ((np.asarray(status).reshape(-1) != 0) & np.isfinite(back).all(axis=1))
        self.diagnostics['bwd_ok'] = int(bwd_ok.sum())
        error = np.max(np.abs(back-old), axis=1)
        good = fwd_ok & bwd_ok & (error < self.config.fb_threshold)
        self.good_point_count = int(good.sum())
        self.diagnostics['roundtrip_ok'] = self.good_point_count
        self.diagnostics['consistent_points'] = self.good_point_count
        if self.good_point_count < self.config.min_inliers:
            return self._failed('insufficient_consistent_points')
        old, new = old[good], new[good]
        self.diagnostics['fb_error'] = float(np.median(error[good]))
        if hint is not None and self.config.center_only:
            expected=np.array([((hint[0]+hint[2])-(self.bbox[0]+self.bbox[2]))/2,
                               ((hint[1]+hint[3])-(self.bbox[1]+self.bbox[3]))/2])
            tolerance=max(2*self.config.ransac_threshold,
                          .1*min(self.bbox[2]-self.bbox[0],self.bbox[3]-self.bbox[1]))
            selected=np.linalg.norm((new-old)-expected,axis=1)<=tolerance
            self.diagnostics.update(motion_hint_used=True,guided_point_count=int(selected.sum()))
            if int(selected.sum())<self.config.min_inliers:
                return self._failed('insufficient_guided_points')
            old,new=old[selected],new[selected]
        matrix, mask = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold, maxIters=300, confidence=.98, refineIters=5)
        origin = np.array(self.roi[:2], dtype=float)
        local_bbox = tuple(v-origin[i%2] for i,v in enumerate(self.bbox))
        quality = assess_motion(matrix, old, new, mask, local_bbox, self.config)
        self.diagnostics['ransac_reject_reason'] = quality['reason'] if not quality['ok'] else ''
        self.diagnostics.update(quality)
        if not quality['ok']:
            return self._failed(quality['reason'])
        if self.config.center_only:
                                                                              
                                                                             
            center = self.quad.mean(axis=0)
            moved_center = transform_points([center-origin], matrix)[0]+origin
            quad = self.quad + (moved_center-center)
        else:
            quad = transform_points(self.quad-origin, matrix)+origin
        self.diagnostics['position_model'] = 'center_translation' if self.config.center_only else 'partial_affine'
        low, high = quad.min(axis=0), quad.max(axis=0)
        height, width = self.shape
        unclipped_area = float(np.prod(high-low))
        low, high = np.maximum(low, [0,0]), np.minimum(high, [width,height])
        if np.any(high-low < 6) or float(np.prod(high-low))/max(unclipped_area, 1) < .65:
            return self._failed('target_out_of_view')
        bbox = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
        points_global = new[np.asarray(mask).reshape(-1).astype(bool)]+origin
        inlier_points = points_global.copy()
        if len(inlier_points):
            self.diagnostics['inlier_centroid'] = [float(v) for v in np.mean(inlier_points, axis=0)]
        inside = (inside_quad(points_global, quad)
                  & (points_global[:,0] >= 0) & (points_global[:,0] < width)
                  & (points_global[:,1] >= 0) & (points_global[:,1] < height))
        points_global = points_global[inside]
        if len(points_global) < self.config.min_inliers:
            return self._failed('points_left_roi')
        x1,y1,x2,y2 = self.roi
        margin = max(18, .25*max(bbox[2]-bbox[0], bbox[3]-bbox[1]))
        fits = bbox[0]-x1 >= margin and bbox[1]-y1 >= margin and x2-bbox[2] >= margin and y2-bbox[3] >= margin
        roi = self.roi if fits else search_roi(bbox, self.shape, self.config)
        current_gray = gray if roi == self.roi else self._gray(frame, roi)
        self.quad, self.bbox = quad, bbox
        self.roi, self.gray = roi, current_gray
        self.points = (points_global-np.array(roi[:2])).reshape(-1,1,2).astype(np.float32)
        self.frames += 1
        if len(self.points) < 40 or self.frames % 10 == 0:
            extra = self._features(self.gray, roi, self.points)
            if extra is not None:
                self.points = np.concatenate((self.points, extra))
        self.last_failure_reason = ''
        return True, bbox
