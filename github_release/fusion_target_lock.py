"""Detector-owned size, relative image motion, and source-time corrections.

The legacy TargetLock remains available for A/B tests. This state machine uses
its adapters/identity helpers, but never requires LK projection to accept YOLO.
"""
import math
import time

from source_aware_fusion import SourceAwareFusion
from visual_target_lock import TargetLock, LockResult, LockedTarget
from template_recovery import TemplateRecovery


class FusionTargetLock(TargetLock):
    def __init__(self, *args, template_bank=None, **kwargs):
        kwargs['use_sort'] = False
        super().__init__(*args, **kwargs)
        if self.prediction_timeout > .35:
            raise ValueError('Source-aware blind prediction must not exceed 0.35 seconds')
        self.fusion = SourceAwareFusion(history_seconds=self.max_detection_age+.1,
                                        prediction_horizon=self.prediction_timeout)
        self._visual_box = self._visual_at = None
        self._last_source = None
        self.detector_box = self.detector_at = None
        self.projection_rejections = 0
        self.template_bank=template_bank
        self.template_recovery=TemplateRecovery(template_bank) if template_bank is not None else None
        self.template_recoveries=0
        self.template_updates=0
        self.template_observed_at=None
        self._require_acquisition_confirmation=False

    def _release(self,forget=False):
        super()._release(forget=forget)
        bank=getattr(self,'template_bank',None)
        recovery=getattr(self,'template_recovery',None)
        if forget:
            if bank is not None:
                bank.reset()
            if recovery is not None:
                recovery.reset()
            if hasattr(self,'_require_acquisition_confirmation'):
                self._require_acquisition_confirmation=True
        elif recovery is not None:
            recovery.clear_confirmation('waiting_for_target')

    def _check_template(self,frame,timestamp,force=False):
        if self.template_recovery is None or self.target is None:
            return None
        position=self.fusion.current(timestamp)
        reference=position.bbox if position else self.box
        return self.template_recovery.check(frame,reference,timestamp,self.detector_at,force=force)

    def _start_tracker(self, frame, bbox):
        super()._start_tracker(frame, bbox)
        self._visual_box = bbox if self.tracker is not None else None
        self._visual_at = self.last_frame_at if self.tracker is not None else None

    def detection_context(self, timestamp):
        reference = self.fusion.reference_at(timestamp)
        return dict(epoch=self.epoch, reference=reference or self.box,
                    last_reference=self.detector_box,
                    track_id=self.active_id or (self.memory['id'] if self.memory else None))

    def _choose(self, detections, source_frame, reference, source_at, width, height, visual_ok):
        valid = [d for d in detections if self._box(d.bbox,width,height) is not None]
        expected = self.target.class_id if self.target else self.memory['class_id'] if self.memory else None
        same = [d for d in valid if expected is None or d.class_id == expected]
        appearance_diagnostics=[]
        if self.target is None and self.memory is None:
            if len(same)!=1:
                return None,False,False,'ambiguous' if len(same)>1 else 'no_detection',appearance_diagnostics
            if self._require_acquisition_confirmation:
                return same[0],True,True,'acquisition_candidate',appearance_diagnostics
            return same[0],False,False,'acquired',appearance_diagnostics

        if self.template_bank is not None and self.template_bank.ready:
            compatible=[]
            continuous=[]
            detector_reanchors=[]
            prior=reference or self.box or self.memory['bbox']
            size=self.detector_box or prior
            reference_area=max(1.,(size[2]-size[0])*(size[3]-size[1]))
            support_age=max(0.,source_at-self.detector_at) if self.detector_at is not None else math.inf
            for item in same:
                score=self.template_bank.verify(source_frame,item.bbox)
                ok=score is not None and score>=self.template_bank.min_match_score
                appearance_diagnostics.append(dict(bbox=list(item.bbox),confidence=item.confidence,
                    class_id=item.class_id,template_score=score,reason='template_match' if ok else 'template_mismatch'))
                if ok:
                    compatible.append(item)
                    continue

                if self.target is not None and visual_ok and not self.needs_recovery:
                                                                               
                                                                              
                                                                         
                    ratio=((item.bbox[2]-item.bbox[0])*(item.bbox[3]-item.bbox[1]) /
                           reference_area)
                    if self._distance(item.bbox,prior,width,height)<=self.gate and .25<=ratio<=4.:
                        continuous.append(item)
                        continue

                                                                           
                                                                         
                distance=self._distance(item.bbox,prior,width,height)
                ratio=((item.bbox[2]-item.bbox[0])*(item.bbox[3]-item.bbox[1]) /
                       reference_area)
                spatial=distance<=self.gate and self._overlap(item.bbox,prior)>=.1
                recovering=self.needs_recovery or not visual_ok or not spatial or not .25<=ratio<=4.
                strong=(recovering and self.single_target_recovery and self.detector_reanchor
                        and item.confidence>=self.detector_reanchor_confidence
                        and (not visual_ok or support_age>=self.detector_reanchor_after))
                if (strong and distance<=self.recovery_gate
                        and 1/self.recovery_area_ratio<=ratio<=self.recovery_area_ratio):
                    detector_reanchors.append(item)

            if compatible:
                same=compatible
            elif continuous:
                same=continuous
            elif detector_reanchors:
                same=detector_reanchors
            elif same:
                return None,False,False,'template_mismatch',appearance_diagnostics
        if len(same) != 1:
            return None, False, False, 'ambiguous' if len(same)>1 else 'no_detection', appearance_diagnostics
        item = same[0]
        reference = reference or self.box or self.memory['bbox']
        distance = self._distance(item.bbox,reference,width,height)
                                                                         
        size_box = self.detector_box or reference
        area_ratio = ((item.bbox[2]-item.bbox[0])*(item.bbox[3]-item.bbox[1]) /
                      ((size_box[2]-size_box[0])*(size_box[3]-size_box[1])))
        spatial = distance <= self.gate and self._overlap(item.bbox,reference)>=.1
        size_ok = .25 <= area_ratio <= 4.
        recovering = self.needs_recovery or not visual_ok or not spatial or not size_ok
        similarity = self._similarity(self._describe(source_frame,item.bbox))
        support_age = max(0., source_at-self.detector_at) if self.detector_at is not None else math.inf
        strong = (recovering and self.single_target_recovery and self.detector_reanchor
                  and item.confidence >= self.detector_reanchor_confidence
                  and (not visual_ok or support_age >= self.detector_reanchor_after))
        reason = 'spatial_match'
        if not self.single_target_recovery and similarity is not None and similarity < .55:
            reason = 'appearance_mismatch'
        elif recovering and not self.single_target_recovery:
            if similarity is None or similarity < self.appearance_threshold:
                reason = 'appearance_mismatch'
            elif not size_ok or distance > self.recovery_gate:
                reason = 'recovery_motion_gate'
        elif recovering and not strong:
            if not size_ok or not 1/self.recovery_area_ratio <= area_ratio <= self.recovery_area_ratio:
                reason = 'size_gate'
            elif distance > self.recovery_gate:
                reason = 'recovery_motion_gate'
        if strong:
            reason = 'detector_reanchor_candidate'
        elif recovering and reason == 'spatial_match':
            reason = 'recovery_candidate'
        diagnostics = [dict(bbox=list(item.bbox),confidence=item.confidence,class_id=item.class_id,
                            center_distance=distance,area_ratio=area_ratio,
                            reference_bbox=list(reference),detector_support_age_s=support_age,
                            appearance_score=similarity,reason=reason)]
        allowed = reason in ('spatial_match','recovery_candidate','detector_reanchor_candidate')
        return (item if allowed else None, recovering, strong,
                reason if allowed or reason=='appearance_mismatch' else 'association_rejected', diagnostics)

    def update(self, frame, timestamp, detect=None, detection=None):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Frame time must be finite and nonnegative')
        height,width = frame.shape[:2]
        if self.last_frame_at is not None and timestamp <= self.last_frame_at:
            raise ValueError('Frame times must increase')
        if self.last_frame_at is not None and (frame.shape[:2] != self.shape or
                                               timestamp-self.last_frame_at > self.max_frame_gap):
            if frame.shape[:2]!=self.shape and self.template_bank is not None:
                self.template_bank.reset()
            if self.template_recovery is not None:
                self.template_recovery.reset()
            self._release(forget=True)
            self.fusion.reset()
            self.detector_box = self.detector_at = None
            self.template_observed_at=None
            self._visual_box = self._visual_at = None
        if self.last_measured_at is not None and timestamp-self.last_measured_at > self.release_timeout:
            self._release()
        if self.memory is not None and timestamp-self.memory['timestamp'] > self.recovery_timeout:
            self.memory = self.pending_recovery = self.pending_reanchor = None
            if self.template_bank is not None:
                self.template_bank.reset()
            if self.template_recovery is not None:
                self.template_recovery.reset()
            self._require_acquisition_confirmation=True
        if self.pending_reanchor and timestamp-self.pending_reanchor[1] > self.recovery_confirm_window:
            self.pending_reanchor = None
        self.shape,self.last_frame_at = frame.shape[:2],timestamp
        self.last_init_ms=0.
        template_verified=False
        template_reset=False
        template_match=self._check_template(frame,timestamp,force=self.tracker is None)
        visual_ok = False
        quality = {}
        failure = ''
        tracking_ms = 0.
        lk_motion_recorded = False
        if self.tracker is not None:
            started=time.perf_counter()
            try:
                if template_match is not None and callable(getattr(self.tracker,'set_motion_hint',None)):
                    self.tracker.set_motion_hint(template_match.bbox)
                ok,box=self.tracker.update(frame)
                quality=dict(getattr(self.tracker,'diagnostics',{}))
                box=self._box(box,width,height) if ok else None
                if box is not None and self._visual_box is not None:
                    dx=((box[0]+box[2])-(self._visual_box[0]+self._visual_box[2]))/2
                    dy=((box[1]+box[3])-(self._visual_box[1]+self._visual_box[3]))/2
                    if math.hypot(2*dx/width,2*dy/height) <= self.gate:
                        reliability=max(.05,min(1.,float(quality.get('inlier_ratio',.5))))
                        self.fusion.record_motion(self._visual_at,timestamp,dx,dy,reliability)
                        lk_motion_recorded = True
                        self._visual_box,self._visual_at=box,timestamp
                        visual_ok=True
                    else:
                        failure='visual_jump_gate'
                if not visual_ok:
                    failure=failure or getattr(self.tracker,'last_failure_reason','visual_update_failed')
            except Exception as error:
                failure='backend_exception'
                self.backend_errors+=1
                self.last_error=str(error)
            tracking_ms=(time.perf_counter()-started)*1000
            if not visual_ok:
                self.tracker=None
                self.needs_recovery=True
                self.visual_failures+=1

        if template_match is None and not visual_ok:
            template_match=self._check_template(frame,timestamp,force=True)
        if template_match is not None:
            verified_box=self._box(template_match.bbox,width,height)
            if verified_box is not None:
                position=self.fusion.current(timestamp)
                needs_image_anchor=(not visual_ok or position is None or position.method=='state_prediction')
                if needs_image_anchor:
                    uncertainty=max(2.,.05*min(verified_box[2]-verified_box[0],verified_box[3]-verified_box[1]))
                    verified=self.fusion.observe_current(verified_box,timestamp,uncertainty)
                    if verified is not None:
                        self.box=verified.bbox
                        self.last_measured_at=timestamp
                        self.needs_recovery=False
                        self._remember(timestamp)
                        self._start_tracker(frame,self.box)
                        self.template_recoveries+=1
                        self.template_observed_at=timestamp
                        template_verified=template_reset=True

        ran=detection is not None or (detect is not None and self.detection_due(timestamp))
        detections,tracks,inference_ms=[],[],None
        accepted=recovered=reanchored=False
        reason,alignment='not_run','none'
        diagnostics=[]
        result_age=None
        proposal=None
        projection=None
        source_at=timestamp
        alignment_ms=0.
        if ran:
            if detection is not None:
                detections,tracks,inference_ms=detection.detections,detection.tracks,detection.inference_ms
                source_at,source_frame=detection.received_at,detection.frame
                result_age=(timestamp-source_at)*1000
                context=detection.context
            else:
                self.mark_detection_requested(timestamp)
                detections,tracks,inference_ms=detect()
                source_frame,context=frame,self.detection_context(timestamp)
            self.detector_calls+=1
            self.detector_hit_frames+=bool(detections)
            if (not math.isfinite(source_at) or not 0 <= timestamp-source_at <= self.max_detection_age
                    or source_frame.shape[:2] != frame.shape[:2]):
                reason='stale_result'
                self.stale_results+=1
            elif context.get('epoch') != self.epoch:
                reason='obsolete_lock_result'
            elif self._last_source is not None and source_at <= self._last_source:
                reason='duplicate_source_result'
            else:
                self._last_source=source_at
                reference=self.fusion.reference_at(source_at) or context.get('reference')
                proposal,recovering,strong,reason,diagnostics=self._choose(
                    detections,source_frame,reference,source_at,width,height,visual_ok or template_verified)
                if reason=='template_mismatch' and visual_ok:
                    position=self.fusion.current(timestamp)
                    score=self.template_bank.verify(frame,position.bbox) if position else None
                    if score is None or score<self.template_bank.min_match_score:
                        visual_ok=False
                        self.tracker=None
                        self.needs_recovery=True
                if reason=='appearance_mismatch' and visual_ok and not self.single_target_recovery:
                    visual_position=self.fusion.current(timestamp)
                    current_similarity=self._similarity(self._describe(
                        frame,visual_position.bbox if visual_position else self.box))
                    if current_similarity is not None and current_similarity < .55:
                        visual_ok=False
                        self.tracker=None
                        self.needs_recovery=True
                        self.appearance_conflicts+=1
                if proposal is not None:
                    if (recovering and self.template_observed_at is not None
                            and source_at<self.template_observed_at):
                                                                               
                                                                              
                                                                            
                        proposal=None
                        reason='newer_image_observation'
                if proposal is not None:
                    restart=recovering or (self.target is None and self.memory is None)
                    started=time.perf_counter()
                    projection=self.fusion.preview(proposal.bbox,source_at,timestamp,
                                                   proposal.confidence,restart_motion=restart)
                    alignment_ms=(time.perf_counter()-started)*1000
                    if projection is None or self._box(projection.bbox,width,height) is None:
                        reason='projection_uncertain'
                        self.projection_rejections+=1
                        proposal=None
                    elif recovering and not self._confirm_reanchor(
                            proposal.bbox, timestamp,source_at,width,height):
                        reason='detector_reanchor_pending' if strong else 'recovery_pending'
                        proposal=None
                    if proposal is not None:
                        projection=self.fusion.commit(proposal.bbox,source_at,timestamp,
                            proposal.confidence,restart_motion=restart)
                        if projection is not None:
                            accepted=True
                            recovered,reanchored=recovering and not strong,strong
                            if self.active_id is None:
                                if self.memory is not None:
                                    self.active_id=self.memory['id']
                                else:
                                    self.counter+=1
                                    self.active_id=self.counter
                                    self.epoch+=1
                            self._require_acquisition_confirmation=False
                            self.target=proposal
                            self.detector_box,self.detector_at=tuple(proposal.bbox),source_at
                            self.last_confirmed_at=source_at
                            self.box=tuple(projection.bbox)
                            self.last_measured_at=projection.observed_at
                            self.confirmed_detection_frames+=1
                            self.needs_recovery=False
                            self._remember(projection.observed_at,self._describe(source_frame,proposal.bbox))
                            self.pending_reanchor=None
                            preserved=False
                            if (self.template_bank is not None and not recovering and visual_ok and self.tracker is not None
                                    and callable(getattr(self.tracker,'reanchor',None))):
                                preserved=self.tracker.reanchor(frame,self.box)
                            if preserved:
                                self._visual_box,self._visual_at=self.box,timestamp
                            else:
                                self._start_tracker(frame,self.box)
                            if self.template_bank is not None:
                                if reanchored:
                                                                            
                                                                                
                                                                            
                                                                             
                                    self.template_bank.reset()
                                    if self.template_recovery is not None:
                                        self.template_recovery.reset()
                                changed=self.template_bank.consider(frame,self.box,proposal.confidence,source_at,
                                    image_current=projection.method!='state_prediction',
                                    stable=visual_ok and not template_verified and quality.get('inlier_ratio',0)>=.7,
                                    recovering=recovering)
                                self.template_updates+=bool(changed)
                            alignment=projection.method
                            if recovering:
                                self.epoch+=1
                            if strong:
                                self.detector_reanchors+=1
                                reason='detector_reanchored'
                            elif recovered:
                                self.identity_recoveries+=1
                                reason='identity_recovered'
                            else:
                                reason='accepted'
                if reason in ('ambiguous','association_rejected','appearance_mismatch','template_mismatch'):
                    self.pending_reanchor=None
            if detections and not accepted:
                self.rejected_results+=1
            self.reason_counts[reason]+=1

        observation=None
        current=projection if accepted else self.fusion.current(timestamp)
        source='none'
        if current is not None and self.target is not None:
            box=self._box(current.bbox,width,height)
            if box is not None:
                                                                           
                                                                             
                image_current=current.method!='state_prediction'
                if image_current and (accepted or visual_ok or template_verified):
                    source='detector' if accepted else 'visual'
                    self.box=box
                    self.last_measured_at=current.observed_at
                    self._remember(current.observed_at)
                    deadline=timestamp+self.max_frame_gap
                elif self.last_measured_at is not None and timestamp-self.last_measured_at <= self.prediction_timeout:
                    source='prediction'
                    deadline=self.last_measured_at+self.prediction_timeout
                if source!='none':
                    observation=LockedTarget(box,self.target.confidence if source=='detector' else 0.,
                        self.target.class_id,self.active_id,source,deadline,recovered or reanchored or template_reset)
                    self.visual_frames+=source=='visual'
                    self.prediction_frames+=source=='prediction'
        if ran:
            self.last_detection_result=dict(source_timestamp=source_at,consumed_timestamp=timestamp,
                target_count=len(detections),accepted=accepted,association_reason=reason,
                position_alignment=alignment,alignment_reason='projection_uncertain' if reason=='projection_uncertain' else '',
                result_age_at_consumption_ms=result_age)
        result=LockResult(observation,detections,tracks,inference_ms,ran,
            ('predicting' if source=='prediction' else 'locked_'+source) if observation else
            ('ambiguous' if reason=='ambiguous' else 'lost_hold' if self.target else 'searching'),
            detector_track_id=proposal.track_id if proposal is not None else None,
            position_age_s=timestamp-self.last_measured_at if self.last_measured_at is not None else None,
            detection_age_s=timestamp-self.detector_at if self.detector_at is not None else None,
            detector_accepted=accepted,detector_result_age_ms=result_age,association_reason=reason,
            association_candidates=diagnostics,recovered=recovered,position_alignment=alignment,
            visual_update_ok=visual_ok,recovery_pending_count=self.pending_reanchor[2] if self.pending_reanchor else 0,
            visual_failure_reason=failure,visual_point_count=quality.get('consistent_points'),
            lk_fwd_ok=quality.get('fwd_ok'),lk_bwd_ok=quality.get('bwd_ok'),
            lk_roundtrip_ok=quality.get('roundtrip_ok'),
            lk_visual_used=lk_motion_recorded,
            lk_ransac_reject_reason=quality.get('ransac_reject_reason','not_run'),
            visual_tracking_ms=tracking_ms,visual_init_ms=self.last_init_ms,alignment_ms=alignment_ms,
            visual_inlier_count=quality.get('inlier_count'),visual_inlier_ratio=quality.get('inlier_ratio'),
            visual_reprojection_error=quality.get('reprojection_error'),visual_fb_error=quality.get('fb_error'),
            visual_scale=quality.get('scale'),visual_rotation_deg=quality.get('rotation_deg'),
            visual_roi_area=quality.get('roi_area'),alignment_method=alignment,
            alignment_reason='projection_uncertain' if reason=='projection_uncertain' else '',
            detector_reanchor_pending_count=self.pending_reanchor[2] if self.pending_reanchor else 0,
            detector_reanchored=reanchored)
        result.fusion_uncertainty_px=current.uncertainty_px if current else None
        result.fusion_observed_at=current.observed_at if current else None
        result.fusion_provenance=current.method if current else 'none'
        if template_verified:
            result.fusion_provenance='template_verified'
        result.template_ready=bool(self.template_bank is not None and self.template_bank.ready)
        result.template_verified=template_verified
        result.template_score=template_match.score if template_match is not None else None
        result.template_peak_margin=template_match.margin if template_match is not None else None
        if (template_match is None and self.template_recovery is not None
                and self.template_recovery.checked_at==timestamp):
            result.template_score=self.template_bank.diagnostics.get('best_score')
            result.template_peak_margin=self.template_bank.diagnostics.get('peak_margin')
        result.template_search_ms=(self.template_bank.diagnostics.get('elapsed_ms',0.)
            if self.template_recovery is not None and self.template_recovery.checked_at==timestamp else 0.)
        result.template_status=self.template_recovery.status if self.template_recovery else 'disabled'
        result.template_confirmations=self.template_recovery.streak if self.template_recovery else 0
        result.flow_motion_hint_used=quality.get('motion_hint_used',False)
        result.flow_guided_points=quality.get('guided_point_count')
        return result

    def summary(self):
        data=super().summary()
        data.update(motion_backend='source_aware_displacement',fusion_mode='source_aware',
                    dimension_authority='detector',projection_rejections=self.projection_rejections,
                    template_enabled=self.template_bank is not None,
                    template_ready=bool(self.template_bank is not None and self.template_bank.ready),
                    template_recoveries=self.template_recoveries,template_updates=self.template_updates)
        return data
