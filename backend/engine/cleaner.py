import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .calibrated_data import get_calibrated_alpha, WATERMARK_COLOR_BGR
from .lama_inpaint import LaMaInpainter

class WatermarkCleaner:
    def __init__(self):
        self.calibrated_alpha_base = get_calibrated_alpha() # 64x64 float32 [0..0.38]
        self.watermark_color = WATERMARK_COLOR_BGR.copy() # [255, 255, 255]
        self.lama = LaMaInpainter()

    def _get_scaled_alpha(self, w: int, h: int) -> np.ndarray:
        """Scales the calibrated alpha mask to target width and height."""
        if (h, w) == self.calibrated_alpha_base.shape:
            return self.calibrated_alpha_base.copy()
        return cv2.resize(
            self.calibrated_alpha_base,
            (w, h),
            interpolation=cv2.INTER_CUBIC if w > 64 else cv2.INTER_AREA
        )

    def _estimate_local_background(self, roi: np.ndarray) -> np.ndarray:
        border_width = max(2, min(6, roi.shape[0] // 8, roi.shape[1] // 8))
        border = np.concatenate([
            roi[:border_width, :].reshape(-1, 3),
            roi[-border_width:, :].reshape(-1, 3),
            roi[:, :border_width].reshape(-1, 3),
            roi[:, -border_width:].reshape(-1, 3)
        ], axis=0)
        return np.median(border, axis=0)

    def _inverse_alpha_restore(
        self,
        roi: np.ndarray,
        alpha: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        bg_local = self._estimate_local_background(roi)
        adaptive_alpha = np.clip(alpha * strength, 0.0, 0.52)
        a3 = np.repeat(adaptive_alpha[:, :, np.newaxis], 3, axis=2)
        denom = np.maximum(1e-4, 1.0 - a3)
        restored = (roi - a3 * self.watermark_color) / denom
        min_floor = np.minimum(roi, bg_local - 12.0)
        restored = np.maximum(restored, min_floor)
        trans = np.clip(adaptive_alpha / 0.045, 0.0, 1.0)
        trans3 = np.repeat(trans[:, :, np.newaxis], 3, axis=2)
        blended = restored * trans3 + roi * (1.0 - trans3)
        return np.clip(blended, 0.0, 255.0)

    def clean(
        self,
        image: np.ndarray,
        bbox: list[int],
        mode: str = "lama",
        custom_mask: Optional[np.ndarray] = None,
        feather: int = 2,
        is_watermark_present: bool = True,
        aggressive: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Removes watermark from image within the specified bbox.
        
        Modes:
        - 'lama': State-of-the-Art Deep Learning Generative Inpainting (Fast Fourier Convolutions) [DEFAULT]
        - 'smart': Adaptive Inverse Alpha Restoration with background-floor protection (Ultra-fast, zero blur)
        - 'hybrid': Smart inverse alpha with bilateral boundary smoothing
        - 'inpaint': OpenCV Telea inpainting
        """
        h_img, w_img = image.shape[:2]
        x, y, w, h = bbox
        
        if not is_watermark_present:
            return image.copy(), np.zeros((h_img, w_img), dtype=np.uint8), {
                "mode": mode,
                "bbox": bbox,
                "modified_pixels": 0,
                "modified_percent": 0.0,
                "status": "untouched"
            }

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        actual_w = x2 - x1
        actual_h = y2 - y1
        
        if actual_w <= 0 or actual_h <= 0:
            return image.copy(), np.zeros((h_img, w_img), dtype=np.uint8), {"modified_pixels": 0}
            
        roi = image[y1:y2, x1:x2].astype(np.float32)
        out_img = image.copy()
        
        # Build precise target mask
        if custom_mask is not None:
            mask_crop = custom_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
            alpha = cv2.GaussianBlur(mask_crop, (3, 3), 0.5) * 0.38
        else:
            scaled_alpha = self._get_scaled_alpha(w, h)
            ox = x1 - x
            oy = y1 - y
            alpha = scaled_alpha[oy:oy + actual_h, ox:ox + actual_w]
            
        alpha_threshold = 0.012 if aggressive else 0.018
        binary_mask_roi = (alpha > alpha_threshold).astype(np.uint8) * 255

        if custom_mask is None:
            bg_local = self._estimate_local_background(roi)
            bright_delta = np.max(roi - bg_local, axis=2)
            expected_zone = cv2.dilate(
                (alpha > 0.004).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            ).astype(bool)
            residue_like = ((bright_delta > (5.0 if aggressive else 8.0)) & expected_zone).astype(np.uint8) * 255
            binary_mask_roi = cv2.bitwise_or(binary_mask_roi, residue_like)

        full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask_roi
        
        # 1. State-of-the-Art Neural Inpainting (LaMa)
        if mode == "lama" and self.lama.is_available():
            # Pre-remove the known white translucent overlay, then ask LaMa to
            # reconstruct only the remaining uncertain edge/residue pixels.
            pre_restored = image.copy()
            restore_strength = 1.12 if aggressive else 1.0
            pre_restored[y1:y2, x1:x2] = self._inverse_alpha_restore(roi, alpha, restore_strength).astype(np.uint8)
            kernel_size = max(3, 3 + 2 * max(1, feather + (1 if aggressive else 0)))
            dilated_full_mask = cv2.dilate(
                full_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            )
            out_img = self.lama.inpaint(pre_restored, dilated_full_mask)
            
        elif mode == "inpaint" or (mode == "lama" and not self.lama.is_available()):
            dilated = cv2.dilate(binary_mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            restored_roi = cv2.inpaint(roi.astype(np.uint8), dilated, 3, cv2.INPAINT_TELEA).astype(np.float32)
            out_img[y1:y2, x1:x2] = restored_roi.astype(np.uint8)
            
        elif mode in ("smart", "hybrid"):
            # Local background estimation from perimeter
            bg_local = self._estimate_local_background(roi)
            
            # Dynamic alpha scaling based on observed contrast
            center = roi[actual_h//4: 3*actual_h//4, actual_w//4: 3*actual_w//4]
            center_mean = np.mean(center, axis=(0,1))
            denom_alpha = np.maximum(1.0, 255.0 - bg_local)
            a_peak_obs = np.clip((center_mean - bg_local) / denom_alpha, 0.05, 0.45)
            scale_factor = float(np.mean(a_peak_obs) / (np.max(alpha) + 1e-5))
            scale_factor = np.clip(scale_factor, 0.65, 1.05)
            adaptive_alpha = alpha * scale_factor
            
            final_roi = self._inverse_alpha_restore(roi, adaptive_alpha, 1.0)
            
            if mode == "hybrid":
                filtered = cv2.bilateralFilter(final_roi.astype(np.uint8), 5, 20, 20).astype(np.float32)
                edge_band = (adaptive_alpha > 0.01) & (adaptive_alpha < 0.10)
                edge3 = np.repeat(edge_band[:, :, np.newaxis], 3, axis=2)
                final_roi = np.where(edge3, filtered, final_roi)
                
            out_img[y1:y2, x1:x2] = np.clip(final_roi, 0, 255).astype(np.uint8)
            
        modified_count = int(np.count_nonzero(full_mask))
        metadata = {
            "mode": mode,
            "bbox": [x1, y1, actual_w, actual_h],
            "modified_pixels": modified_count,
            "modified_percent": round(100.0 * modified_count / (w_img * h_img), 3)
        }
        
        return out_img, full_mask, metadata
