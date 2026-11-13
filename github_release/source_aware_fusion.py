"""Bounded source-time detector corrections with correlated displacement replay.

Boxes are xyxy. Motion controls never reduce uncertainty or change box size.
All timestamps are nonnegative monotonic-clock seconds.

This is detector-anchor correction plus displacement replay, not a calibrated
Kalman covariance model. sigma is a conservative engineering error budget:
fully correlated flow drift is accumulated linearly in elapsed time. Width and
height belong exclusively to accepted detector observations. A missing motion
interval marks that anchor's projection as prediction. A new detector anchor
with full motion coverage, or separately validated current-image evidence, can
restore image freshness without advancing the detector source clock.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Projection:
    bbox: tuple
    uncertainty_px: float
    observed_at: float
    method: str
    source_at: float

class SourceAwareFusion:
    def __init__(self, history_seconds=.7, max_events=128, prediction_horizon=.35):
        if not all(math.isfinite(v) for v in (history_seconds,max_events,prediction_horizon)) or history_seconds <= 0 or max_events < 1 or int(max_events)!=max_events or prediction_horizon < 0:
            raise ValueError('Invalid history bounds')
        self.history_seconds=float(history_seconds)
        self.max_events=int(max_events)
        self.prediction_horizon=float(prediction_horizon)
        self.max_uncertainty_fraction=.75
        self.reset()

    def reset(self):
        self._motions=[]
        self._states=[]
        self._observations=[]
        self._latest=0.
        self._reference_floor=0.
        self._anchor=None
        self._last_detector=None
        self._last_source=-1.
        self._last_motion_end=-1.

    def _trim(self):
        cutoff=self._latest-self.history_seconds
        keep=[m for m in self._motions if m[1]>=cutoff][-self.max_events:]
        removed_count=len(self._motions)-len(keep)
        if removed_count:
            self._reference_floor=max(self._reference_floor,self._motions[removed_count-1][1])
        observations=[o for o in self._observations if o[0]>=cutoff][-self.max_events:]
        removed_observations=len(self._observations)-len(observations)
        if removed_observations:
            self._reference_floor=max(self._reference_floor,self._observations[removed_observations-1][0])
        boundary=max(self._reference_floor,keep[0][0] if keep else self._latest)
        if self._anchor is not None and boundary>self._anchor[0]:
            self._anchor=self._advance(self._anchor,boundary)
        self._motions=keep
        self._observations=observations
        self._states=sorted((s for s in self._states if s[0]>=max(cutoff,self._reference_floor)),key=lambda s:s[0])[-self.max_events:]

    def record_motion(self,start_at,end_at,dx,dy,quality=1.):
        vals=(start_at,end_at,dx,dy,quality)
        if not all(math.isfinite(v) for v in vals) or start_at<0 or end_at<=start_at or start_at<self._last_motion_end or not 0<=quality<=1:
            raise ValueError('Motion must be finite, ordered and nonoverlapping')
        self._motions.append(tuple(float(v) for v in vals))
        self._last_motion_end=end_at
        self._latest=max(self._latest,end_at)
        self._trim()

    def _advance(self,state,t,motions=None):
        if motions is not None:
            return self._advance_motion(state,t,motions)
                                                                               
                                                                               
                                                                               
        for stamp,cx,cy,sigma in self._observations:
            if state[0] < stamp <= t:
                state=self._advance_motion(state,stamp)
                at,box,vx,vy,_,_,_,_=state
                w,h=box[2]-box[0],box[3]-box[1]
                state=(stamp,(cx-w/2,cy-h/2,cx+w/2,cy+h/2),vx,vy,sigma,stamp,0.,True)
        return self._advance_motion(state,t)

    def _advance_motion(self,state,t,motions=None):
        at,box,vx,vy,sigma,source,gap,covered=state
        x,y,r,b=box; dx=dy=0.; cursor=at
        for start,end,mx,my,q in (self._motions if motions is None else motions):
            if end<=at or start>=t: continue
            a=max(at,start); z=min(t,end)
            if a>cursor:
                dt=a-cursor; gap+=dt; dx+=vx*dt; dy+=vy*dt
                sigma+=max(10.,min(r-x,b-y))*dt*2; covered=False
            dt=z-a
            if q>0:
                dx+=mx*dt/(end-start); dy+=my*dt/(end-start)
                sigma+=dt*(2.+12.*(1.-q))
            else:
                gap+=dt; covered=False; dx+=vx*dt; dy+=vy*dt
                sigma+=max(10.,min(r-x,b-y))*dt*2
            cursor=z
        if cursor<t:
            dt=t-cursor; gap+=dt; covered=False; dx+=vx*dt; dy+=vy*dt
            sigma+=max(10.,min(r-x,b-y))*dt*2
        return (t,(x+dx,y+dy,r+dx,b+dy),vx,vy,sigma,source,gap,covered)

    def _project(self,state,t,motions=None):
        if t<state[0]: return None
        advanced=self._advance(state,t,motions)
        at,box,vx,vy,sigma,source,gap,covered=advanced
        if not all(math.isfinite(v) for v in (*box,sigma)) or gap>self.prediction_horizon+1e-9 or sigma>min(box[2]-box[0],box[3]-box[1])*self.max_uncertainty_fraction: return None
        return Projection(box,sigma,t if covered else source,'current_frame' if t==source else ('motion_replay' if covered else 'state_prediction'),source)

    def _candidate(self,bbox,source_at,current_at,confidence,restart_motion=False):
        try:
            box=tuple(float(v) for v in bbox)
            if len(box)!=4 or not all(math.isfinite(v) for v in (*box,source_at,current_at,confidence)): return None
            if source_at<0 or source_at<=self._last_source or current_at<source_at or source_at<max(current_at,self._latest)-self.history_seconds or not 0<confidence<=1: return None
            if box[2]<=box[0] or box[3]<=box[1]: return None
        except (TypeError,ValueError): return None
        vx=vy=0.
        if self._last_detector is not None and not restart_motion:
            old=self._last_detector; dt=source_at-old[0]
            vx=((box[0]+box[2])-(old[1][0]+old[1][2]))/(2*dt)
            vy=((box[1]+box[3])-(old[1][1]+old[1][3]))/(2*dt)
        sigma=max(1.,min(box[2]-box[0],box[3]-box[1])*.03)/math.sqrt(confidence)
        if not all(math.isfinite(v) for v in (vx,vy,sigma)) or max(abs(vx),abs(vy))>100000: return None
        return (float(source_at),box,vx,vy,sigma,float(source_at),0.,True)

    def preview(self,bbox,source_at,current_at,confidence=1.,restart_motion=False):
        state=self._candidate(bbox,source_at,current_at,confidence,restart_motion)
        return self._project(state,current_at,[] if restart_motion else None) if state else None

    def commit(self,bbox,source_at,current_at,confidence=1.,restart_motion=False):
        state=self._candidate(bbox,source_at,current_at,confidence,restart_motion)
        projection=self._project(state,current_at,[] if restart_motion else None) if state else None
        if projection is None: return None
        if restart_motion:
            self._motions=[]
            self._states=[]
            self._observations=[]
            self._reference_floor=source_at
        self._anchor=state; self._last_detector=state
        self._states.append(state); self._last_source=source_at
        self._latest=max(self._latest,current_at); self._trim()
        return projection

    def observe_current(self,bbox,timestamp,uncertainty_px):
        """Commit caller-validated image evidence, never a predicted box.

        Does not advance detector timestamps or create a target. Callers must
        independently enforce identity, appearance and semantic support limits.
        """
        try:
            box=tuple(float(v) for v in bbox)
            if (self._anchor is None or self._last_detector is None or len(box)!=4
                    or not all(math.isfinite(v) for v in (*box,timestamp,uncertainty_px))
                    or timestamp<self._latest or uncertainty_px<=0
                    or (self._observations and timestamp<=self._observations[-1][0])
                    or box[2]<=box[0] or box[3]<=box[1]):
                return None
        except (TypeError,ValueError):
            return None
        current=self._advance(self._anchor,timestamp)
        detector_box=self._last_detector[1]
        w,h=detector_box[2]-detector_box[0],detector_box[3]-detector_box[1]
        if uncertainty_px>min(w,h)*self.max_uncertainty_fraction:
            return None
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        measured=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
        self._anchor=(timestamp,measured,current[2],current[3],float(uncertainty_px),timestamp,0.,True)
        self._observations.append((timestamp,cx,cy,float(uncertainty_px)))
        self._states.append(self._anchor)
        self._latest=timestamp
        self._trim()
        return self.current(timestamp)

    def current(self,timestamp):
        if not math.isfinite(timestamp) or timestamp<0 or self._anchor is None: return None
        return self._project(self._anchor,timestamp)

    def reference_at(self,timestamp):
        if not math.isfinite(timestamp) or timestamp<max(self._reference_floor,self._latest-self.history_seconds): return None
        states=[s for s in self._states if s[0]<=timestamp]
        if self._anchor is not None and self._anchor[0]<=timestamp: states.append(self._anchor)
        state=max(states,key=lambda s:s[0]) if states else None
        p=self._project(state,timestamp) if state else None
        return p.bbox if p else None
