"""Conservative, bounded appearance memory for supplementing local motion."""
from dataclasses import dataclass
import time
import numpy as np

@dataclass(frozen=True)
class TemplateMatch:
    bbox: tuple
    score: float
    margin: float
    elapsed_ms: float

class ProtectedTemplateBank:
    min_seed_confidence = .65
    min_match_score = .72
    min_anchor_score = .45
    min_peak_margin = .06
    seed_window = 1.0
    scales = (.9, 1., 1.1)

    def __init__(self, cv2, clock=time.perf_counter, max_templates=3, max_search_side=256, budget_ms=12.0):
        for name, value, lower, upper in (
            ('max_templates', max_templates, 1, 3),
            ('max_search_side', max_search_side, 16, 256),
        ):
            if not np.isfinite(value) or int(value) != value or not lower <= value <= upper:
                raise ValueError(name + ' outside supported integer bounds')
        if not np.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError('budget_ms must be positive and finite')
        self.cv2, self.clock = cv2, clock
        self.max_templates = max(1,min(3,int(max_templates)))
        self.max_search_side = max(16,int(max_search_side))
        self.budget_ms = float(budget_ms)
        self.reset()

    @property
    def ready(self):
        return bool(self._templates)

    def reset(self):
        self._templates=[]
        self._pending=None
        self._last_at=None
        self.diagnostics={"seed_reject_reason":"unseeded", "anchor_score":None,
                          "best_score":None, "peak_margin":None,
                          "search_timed_out":False}

    def _resize(self,a,size):
        return self.cv2.resize(a,size,interpolation=getattr(self.cv2,'INTER_AREA',3))

    def _crop(self,frame,bbox):
        try:
            b=np.asarray(bbox,dtype=float)
            if b.shape!=(4,) or not np.isfinite(b).all(): return None
            x,y,r,s=np.rint(b).astype(int)
            if x<0 or y<0 or r>frame.shape[1] or s>frame.shape[0] or r-x<8 or s-y<8: return None
            a=np.asarray(frame[y:s,x:r],dtype=np.float32)
            if a.ndim==3: a=a[...,:3].mean(axis=2)
            if a.ndim!=2 or not np.isfinite(a).all() or a.std()<5: return None
            factor = min(1., 96 / max(a.shape))
            if factor < 1:
                a = self._resize(a, (max(8, round(a.shape[1]*factor)), max(8, round(a.shape[0]*factor))))
            return a.copy()
        except (ValueError,TypeError,AttributeError): return None

    def _score(self,a,b):
        a=self._resize(a,(32,32)).astype(float).ravel()
        b=self._resize(b,(32,32)).astype(float).ravel()
        a-=a.mean(); b-=b.mean()
        return float(np.dot(a,b)/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b)))

    def verify(self,frame,bbox):
        if not self.ready:
            self.diagnostics["anchor_score"] = None
            return None
        a=self._crop(frame,bbox)
        if a is None:
            self.diagnostics["anchor_score"] = 0.
            return 0.
                                                                                     
        anchor_score=self._score(a, self._templates[0])
        self.diagnostics["anchor_score"] = anchor_score
        if anchor_score < self.min_anchor_score:
            return 0.
        return max(0.,max(self._score(a,t) for t in self._templates))

    def consider(self,frame,bbox,confidence,source_at,*,image_current,stable,recovering=False):
        def reject(reason):
            self.diagnostics.update(seed_reject_reason=reason,
                                    seed_confidence=confidence,
                                    seed_source_timestamp=source_at)
            self._pending=None
            return False
        if not image_current:
            return reject('not_detector_image')
        if not stable or recovering:
            return reject('track_not_stable_or_recovering')
        if not np.isfinite(confidence) or confidence<self.min_seed_confidence:
            return reject('seed_confidence_low')
        if not np.isfinite(source_at) or source_at < 0:
            return reject('invalid_source_timestamp')
        if self._last_at is not None and source_at<=self._last_at:
            return reject('nonmonotonic_source_timestamp')
        self._last_at=source_at
        a=self._crop(frame,bbox)
        if a is None: return reject('invalid_or_low_texture_crop')
        if self.ready:
            anchor_score=self._score(a, self._templates[0])
            best_score=max(self._score(a,t) for t in self._templates)
            self.diagnostics.update(anchor_score=anchor_score,best_score=best_score)
            if anchor_score < self.min_anchor_score: return reject('anchor_similarity_low')
            if best_score < self.min_match_score: return reject('template_similarity_low')
            if self.max_templates>1:
                self._templates.append(a)
                self._templates[1:]=self._templates[1:][-(self.max_templates-1):]
                self.diagnostics['seed_reject_reason']='template_added'
                return True
            self.diagnostics['seed_reject_reason']='template_bank_full'
            return False
        old=self._pending
        self._pending=(source_at,a, np.asarray(bbox)[2:] - np.asarray(bbox)[:2])
        if old is None:
            self.diagnostics['seed_reject_reason']='waiting_for_second_seed'
            return False
        if source_at-old[0]>self.seed_window:
            self.diagnostics['seed_reject_reason']='seed_window_expired'
            return False
        ratio=(np.asarray(bbox)[2:] - np.asarray(bbox)[:2])/old[2]
        seed_similarity=self._score(a,old[1])
        self.diagnostics.update(seed_similarity=seed_similarity,
                                seed_size_ratio=[float(v) for v in ratio])
        if np.any(abs(ratio-1)>.2): return reject('seed_size_inconsistent')
        if seed_similarity<self.min_match_score: return reject('seed_similarity_low')
        self._templates=[old[1].copy()]
        if self.max_templates>1: self._templates.append(a)
        self._pending=None
        self.diagnostics['seed_reject_reason']='seeded'
        return True

    def search(self,frame,reference_bbox):
        start=self.clock()
        def expired(): return (self.clock()-start)*1000>self.budget_ms
        def finish(reason,score=0.,margin=0.):
            self.diagnostics.update(reason=reason,elapsed_ms=(self.clock()-start)*1000,
                                    best_score=score,peak_margin=margin,
                                    search_timed_out=reason=='timeout')
        if not self.ready: finish('unseeded'); return None
        try:
            x,y,r,b=map(float,reference_bbox)
            if not np.isfinite([x,y,r,b]).all() or r-x<8 or b-y<8: raise ValueError()
            if x < 0 or y < 0 or r > frame.shape[1] or b > frame.shape[0]: raise ValueError()
            w,h=r-x,b-y; cx,cy=(x+r)/2,(y+b)/2
            left=max(0,int(cx-1.25*w)); top=max(0,int(cy-1.25*h))
            right=min(frame.shape[1],int(cx+1.25*w)); bottom=min(frame.shape[0],int(cy+1.25*h))
            region=np.asarray(frame[top:bottom,left:right],dtype=np.float32)
            if region.ndim==3: region=region[...,:3].mean(axis=2)
            if region.ndim != 2 or not region.size or not np.isfinite(region).all() or region.std() < 5:
                finish('invalid'); return None
            factor=min(1.,self.max_search_side/max(region.shape))
            region=self._resize(region,(max(1,int(region.shape[1]*factor)),max(1,int(region.shape[0]*factor))))
            candidates=[]
            for template in self._templates:
                for scale in self.scales:
                    if expired(): finish('timeout'); return None
                    tw,th=max(8,int(w*factor*scale)),max(8,int(h*factor*scale))
                    if th>region.shape[0] or tw>region.shape[1]: continue
                    t=self._resize(template,(tw,th))
                    response=self.cv2.matchTemplate(region,t,self.cv2.TM_CCOEFF_NORMED)
                    if expired(): finish('timeout'); return None
                    response=np.nan_to_num(response,nan=-1.,posinf=-1.,neginf=-1.)
                    iy,ix=np.unravel_index(np.argmax(response),response.shape)
                    score=float(response[iy,ix]); center=(left+(ix+tw/2)/factor,top+(iy+th/2)/factor)
                    mask=response.copy()
                    yy,xx=np.ogrid[:mask.shape[0],:mask.shape[1]]
                    mask[(abs(xx-ix)<tw*.6)&(abs(yy-iy)<th*.6)]=-1
                    candidates.append((score,center,float(mask.max())))
            if not candidates: finish('no_comparison'); return None
            best=max(candidates,key=lambda c:c[0]); score,center,secondary=best
            for other in candidates:
                if abs(other[1][0]-center[0])>w*.6 or abs(other[1][1]-center[1])>h*.6: secondary=max(secondary,other[0])
            margin=score-secondary
            bbox=(center[0]-w/2,center[1]-h/2,center[0]+w/2,center[1]+h/2)
            if score<self.min_match_score or margin<self.min_peak_margin or (self.verify(frame,bbox) or 0)<self.min_match_score:
                finish('ambiguous_or_weak',score,margin); return None
            if expired(): finish('timeout',score,margin); return None
            finish('matched',score,margin)
            return TemplateMatch(bbox,score,margin,self.diagnostics['elapsed_ms'])
        except (ValueError,TypeError,AttributeError):
            finish('invalid'); return None
