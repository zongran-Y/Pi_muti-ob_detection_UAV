"""Temporal validation of local appearance matches; no detector-age refresh."""
import math


class TemplateRecovery:
    def __init__(self,bank,max_bridge_seconds=.75,interval=.12):
        self.bank=bank
        self.max_bridge_seconds=max_bridge_seconds
        self.interval=interval
        self.reset()

    def reset(self):
        self.checked_at=None
        self.clear_confirmation('unseeded')

    def clear_confirmation(self,status=None):
        """Discard temporal votes while preserving the trusted template bank."""
        self.previous=None
        self.streak=0
        self.candidate=None
        if status is not None:
            self.status=status

    def check(self,frame,reference,timestamp,detector_at,force=False):
        self.candidate=None
        if (not self.bank.ready or reference is None or detector_at is None
                or not math.isfinite(timestamp) or not math.isfinite(detector_at)
                or not 0<=timestamp-detector_at<=self.max_bridge_seconds):
            status='unseeded' if not self.bank.ready else 'semantic_support_expired'
            self.clear_confirmation(status)
            return None
        if self.checked_at is not None:
            if timestamp<=self.checked_at or (not force and timestamp-self.checked_at<self.interval):
                return None
        self.checked_at=timestamp
        match=self.bank.search(frame,reference)
        if match is None:
            self.clear_confirmation(self.bank.diagnostics.get('reason','no_match'))
            return None
        box=match.bbox
        count=1
        if self.previous is not None:
            stamp,old,old_count=self.previous
            dx=((box[0]+box[2])-(old[0]+old[2]))/2
            dy=((box[1]+box[3])-(old[1]+old[3]))/2
            tolerance=max(12.,.5*min(box[2]-box[0],box[3]-box[1]))
            if 0<timestamp-stamp<=.25 and math.hypot(dx,dy)<=tolerance:
                count=old_count+1
        self.previous=(timestamp,box,count)
        self.streak=count
        self.candidate=match
        self.status='verified' if count>=2 else 'confirming'
        return match if count>=2 else None
