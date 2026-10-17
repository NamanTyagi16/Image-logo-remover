import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class LaMaInpainter:
    _instance = None

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = str(base_dir / "models" / "lama_fp32.onnx")

        self.model_path = model_path
        self.session = None
        self.active_provider = None
        self._init_lock = threading.Lock()

    def _select_providers(self):
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        preferred = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [provider for provider in preferred if provider in available]

    def _lazy_init(self):
        if self.session is not None:
            return

        with self._init_lock:
            if self.session is not None:
                return

            if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 190 * 1024 * 1024:
                try:
                    from scripts.download_model import download_model
                    print("[LaMa] Model file missing. Initiating auto-download...")
                    download_model(Path(self.model_path))
                except Exception as exc:
                    print(f"[LaMa] Auto-download error: {exc}")

            if not os.path.exists(self.model_path):
                print(f"[LaMa] Model not found at {self.model_path}. Falling back to Telea inpainting.")
                return

            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            providers = self._select_providers() or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=providers,
            )
            active = self.session.get_providers()
            self.active_provider = active[0] if active else providers[0]
            print(f"LaMa ONNX provider: {self.active_provider}")
            if self.active_provider != "CPUExecutionProvider":
                self._verify_active_provider()

    def _force_cpu_session(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.active_provider = "CPUExecutionProvider"
        print("LaMa ONNX provider fallback: CPUExecutionProvider")

    def _verify_active_provider(self):
        test_image = np.zeros((1, 3, 512, 512), dtype=np.float32)
        test_mask = np.zeros((1, 1, 512, 512), dtype=np.float32)
        try:
            self.session.run(None, {"image": test_image, "mask": test_mask})
        except Exception as exc:
            print(f"LaMa provider {self.active_provider} failed compatibility check: {exc}")
            self._force_cpu_session()

    def is_available(self) -> bool:
        return os.path.exists(self.model_path)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Runs LaMa inpainting on the image guided by binary mask (255 for target area).
        Extracts a 512x512 contextual window containing the mask, infers, and re-inserts.
        """
        self._lazy_init()
        if self.session is None:
            dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            return cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)

        h_img, w_img = image.shape[:2]

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return image.copy()

        bx_min, bx_max = int(x_indices.min()), int(x_indices.max())
        by_min, by_max = int(y_indices.min()), int(y_indices.max())
        cx = (bx_min + bx_max) // 2
        cy = (by_min + by_max) // 2

        crop_size = 512
        crop_x1 = max(0, min(w_img - crop_size, cx - crop_size // 2))
        crop_y1 = max(0, min(h_img - crop_size, cy - crop_size // 2))
        crop_x2 = min(w_img, crop_x1 + crop_size)
        crop_y2 = min(h_img, crop_y1 + crop_size)

        pad_bottom = max(0, crop_size - (crop_y2 - crop_y1))
        pad_right = max(0, crop_size - (crop_x2 - crop_x1))

        crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        if pad_bottom > 0 or pad_right > 0:
            crop_img = cv2.copyMakeBorder(crop_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
            crop_mask = cv2.copyMakeBorder(crop_mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)

        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_f = (crop_mask.astype(np.float32) > 0).astype(np.float32)

        img_tensor = np.expand_dims(img_rgb.transpose(2, 0, 1), axis=0)
        mask_tensor = np.expand_dims(np.expand_dims(mask_f, axis=0), axis=0)

        try:
            out_raw = self.session.run(None, {"image": img_tensor, "mask": mask_tensor})[0]
        except Exception:
            if self.active_provider != "CPUExecutionProvider":
                self._force_cpu_session()
                out_raw = self.session.run(None, {"image": img_tensor, "mask": mask_tensor})[0]
            else:
                raise

        out_patch = np.clip(out_raw[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_patch, cv2.COLOR_RGB2BGR)

        actual_h = crop_y2 - crop_y1
        actual_w = crop_x2 - crop_x1
        out_bgr = out_bgr[:actual_h, :actual_w]
        actual_mask = crop_mask[:actual_h, :actual_w]

        dilated_mask = cv2.dilate(actual_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        feathered = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (5, 5), 1.5)
        feather3 = np.repeat(feathered[:, :, np.newaxis], 3, axis=2)

        original_patch = image[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
        blended_patch = out_bgr.astype(np.float32) * feather3 + original_patch * (1.0 - feather3)

        result = image.copy()
        result[crop_y1:crop_y2, crop_x1:crop_x2] = np.clip(blended_patch, 0, 255).astype(np.uint8)
        return result
