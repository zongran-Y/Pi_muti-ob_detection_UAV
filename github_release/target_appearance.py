"""Small, CPU-only appearance descriptors; not a trained identity/ReID model."""


class PatchAppearance:
    def __init__(self, cv2):
        import numpy as np
        self.cv2, self.np = cv2, np

    def describe(self, frame, bbox):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (round(v) for v in bbox)
        patch = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if min(patch.shape[:2]) < 6:
            return None
        patch = cv2.resize(patch, (24, 24))
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1)
        if float(gray.std()) < 5.0:
            return None                                                             
        gray -= gray.mean()
        gray /= max(float(np.linalg.norm(gray)), 1e-8)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
        hist = hist / max(float(hist.sum()), 1e-8)
        return gray, hist

    def similarity(self, first, second):
        np = self.np
        correlation = max(0.0, min(1.0, float(np.dot(first[0], second[0]))))
        color = max(0.0, min(1.0, float(np.sqrt(first[1] * second[1]).sum())))
        return .75 * correlation + .25 * color


def create_box_projector(cv2, config=None, history_seconds=.7, max_steps=3, budget_ms=40, legacy=False):
    """Locate an old detection in the current image, never just copy its box."""
    if not legacy:
        from buffered_alignment import BufferedBoxProjector
        return BufferedBoxProjector(cv2, config, history_seconds=history_seconds,
                                    max_steps=max_steps, budget_ms=budget_ms)
    from visual_target_lock import _LegacyFlowTracker

    def project(source_frame, bbox, current_frame):
        tracker = _LegacyFlowTracker(cv2)
        if not tracker.init(source_frame, bbox):
            return None
        ok, projected = tracker.update(current_frame)
        return projected if ok else None

    return project
