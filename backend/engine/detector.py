import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .calibrated_data import (
    get_calibrated_template,
    get_calibrated_alpha,
    NOMINAL_CENTER_X_REL,
    NOMINAL_CENTER_Y_REL,
    NOMINAL_BOX_SIZE_REL,
    BASE_HEIGHT,
    BASE_WIDTH,
)

@dataclass
class DetectionResult:
    detected: bool
    is_present: bool      # True if watermark actually exists in image
    bbox: List[int]        # [x, y, w, h]
    center: Tuple[float, float] # (cx, cy)
    confidence: float     # 0.0 to 1.0
    flagged: bool         # True if confidence is borderline or low
    scale: float          # estimated scale relative to 768p
    message: str          # Status or diagnostic message

class WatermarkDetector:
    def __init__(self):
        self.base_template = get_calibrated_template() # 64x64 float32 [0..1]
        self.base_template_uint8 = (self.base_template * 255.0).astype(np.uint8)
        self.alpha_mask = get_calibrated_alpha()
        
    def _compute_dog(self, gray: np.ndarray, ksize1: int = 3, sigma1: float = 1.0,
                     ksize2: int = 9, sigma2: float = 2.5) -> np.ndarray:
        """Difference of Gaussians bandpass filter to highlight watermark edges."""
        g1 = cv2.GaussianBlur(gray, (ksize1, ksize1), sigma1)
        g2 = cv2.GaussianBlur(gray, (ksize2, ksize2), sigma2)
        return cv2.subtract(g1, g2)

    def detect(self, image: np.ndarray, sensitivity: float = 1.0) -> DetectionResult:
        """
        Detects the recurring watermark in the bottom-right quadrant of the image.
        Supports variable resolutions, slight positional offsets, and scale variations.
        """
        h_img, w_img = image.shape[:2]
        
        # Base scale calculation relative to 768p
        scale_nominal = float(h_img) / BASE_HEIGHT
        box_size_nominal = int(round(64.0 * scale_nominal))
        
        # Nominal expected center
        nom_cx = int(round(w_img * NOMINAL_CENTER_X_REL))
        nom_cy = int(round(h_img * NOMINAL_CENTER_Y_REL))
        
        nom_x = max(0, nom_cx - box_size_nominal // 2)
        nom_y = max(0, nom_cy - box_size_nominal // 2)
        nom_bbox = [int(nom_x), int(nom_y), int(min(box_size_nominal, w_img - nom_x)), int(min(box_size_nominal, h_img - nom_y))]
        
        # Define search region in bottom-right quadrant. Real exports can drift
        # slightly after resizing/cropping, so keep the anchor but allow room.
        search_radius_x = int(round(115 * scale_nominal))
        search_radius_y = int(round(115 * scale_nominal))
        
        x_min = max(int(w_img * 0.58), nom_cx - search_radius_x)
        x_max = min(w_img, nom_cx + search_radius_x)
        y_min = max(int(h_img * 0.55), nom_cy - search_radius_y)
        y_max = min(h_img, nom_cy + search_radius_y)
        
        roi = image[y_min:y_max, x_min:x_max]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return DetectionResult(
                detected=False,
                is_present=False,
                bbox=nom_bbox,
                center=(float(nom_cx), float(nom_cy)),
                confidence=0.0,
                flagged=False,
                scale=float(scale_nominal),
                message="No search area"
            )
            
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        roi_dog = self._compute_dog(roi_gray)
        
        # Multi-scale search. A narrow scale band was the main reason some
        # compressed or resized samples landed a few pixels off.
        scale_factors = np.arange(0.76, 1.31, 0.04)
        scales_to_test = [float(f * scale_nominal) for f in scale_factors]
        
        best_score = -1.0
        best_loc = (0, 0)
        best_scale = scale_nominal
        best_box_size = box_size_nominal
        
        for sc in scales_to_test:
            cur_size = max(16, int(round(64.0 * sc)))
            if cur_size >= roi.shape[0] or cur_size >= roi.shape[1]:
                continue
            
            tmpl_resized = cv2.resize(self.base_template_uint8, (cur_size, cur_size), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_CUBIC)
            tmpl_dog = self._compute_dog(tmpl_resized)
            
            res = cv2.matchTemplate(roi_dog, tmpl_dog, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            cand_cx = float(x_min + max_loc[0] + cur_size / 2.0)
            cand_cy = float(y_min + max_loc[1] + cur_size / 2.0)
            dist = float(np.hypot(cand_cx - nom_cx, cand_cy - nom_cy) / max(1.0, scale_nominal * 64.0))
            anchor_bonus = 0.06 * max(0.0, 1.0 - 0.35 * dist)
            score = float(max_val) + anchor_bonus

            if score > best_score:
                best_score = score
                best_loc = (int(max_loc[0]), int(max_loc[1]))
                best_scale = float(sc)
                best_box_size = int(cur_size)
                
        det_x = int(x_min + best_loc[0])
        det_y = int(y_min + best_loc[1])
        det_cx = float(det_x + best_box_size / 2.0)
        det_cy = float(det_y + best_box_size / 2.0)

        used_nominal_fallback = False

        # Partial star matches score well on line-art backgrounds but leave the
        # outer sparkle untouched. If the best match is clearly undersized, use
        # the known export anchor instead of the partial-match center.
        min_full_box_size = max(box_size_nominal, int(round(64.0 * scale_nominal)))
        if best_box_size < int(round(min_full_box_size * 0.94)):
            used_nominal_fallback = True
            det_x = nom_bbox[0]
            det_y = nom_bbox[1]
            best_box_size = min_full_box_size
            best_scale = scale_nominal
            det_cx = float(det_x + best_box_size / 2.0)
            det_cy = float(det_y + best_box_size / 2.0)
        
        # Check presence:
        # Check standard deviation and contrast in candidate ROI
        cand_roi = image[max(0, det_y):min(h_img, det_y + best_box_size), max(0, det_x):min(w_img, det_x + best_box_size)]
        cand_gray = cv2.cvtColor(cand_roi, cv2.COLOR_BGR2GRAY) if cand_roi.ndim == 3 else cand_roi
        roi_std = float(np.std(cand_gray)) if cand_gray.size > 0 else 0.0
        
        # If candidate region is virtually uniform flat white/paper and match score is low
        is_flat_clean = (roi_std < 3.5) and (best_score < 0.15)
        
        dist_from_nom = float(np.hypot(det_cx - nom_cx, det_cy - nom_cy) / (scale_nominal * 64.0))
        anchor_score = max(0.0, 1.0 - 0.25 * dist_from_nom)
        
        raw_conf = (max(0.0, best_score) / 0.50) * 0.75 + (anchor_score * 0.25)
        confidence = float(np.clip(raw_conf, 0.0, 1.0))
        
        conf_threshold = 0.16 / max(0.2, sensitivity)
        flag_threshold = 0.30 / max(0.2, sensitivity)
        
        detected = bool((best_score >= conf_threshold) and (dist_from_nom < 2.25) and not is_flat_clean)
        is_present = not is_flat_clean
        flagged = bool((confidence < flag_threshold) or not detected)
        
        final_bbox = [int(det_x), int(det_y), int(best_box_size), int(best_box_size)] if detected else nom_bbox
        final_center = (float(det_cx), float(det_cy)) if detected else (float(nom_cx), float(nom_cy))
        
        if not is_present:
            msg = "Clean / Watermark-free (Untouched)"
        elif used_nominal_fallback and detected and not flagged:
            msg = "Confident detection (anchored full mask)"
        elif used_nominal_fallback and detected:
            msg = "Anchored full mask (Review suggested)"
        elif detected and not flagged:
            msg = "Confident detection"
        elif detected:
            msg = "Borderline detection (Review suggested)"
        else:
            msg = "Logo not confidently identified (Review suggested)"
            
        return DetectionResult(
            detected=detected,
            is_present=is_present,
            bbox=final_bbox,
            center=final_center,
            confidence=round(confidence, 3),
            flagged=flagged,
            scale=round(best_scale, 3),
            message=msg
        )
