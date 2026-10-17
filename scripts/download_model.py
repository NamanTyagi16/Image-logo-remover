import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"
MODEL_PATH = MODEL_DIR / "lama_fp32.onnx"
MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MIN_BYTES = 190 * 1024 * 1024  # ~208 MB model

def download_model(target_path: Path = MODEL_PATH) -> bool:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if target_path.exists() and target_path.stat().st_size >= MIN_BYTES:
        print(f"[Model Check] Model already exists at {target_path} ({target_path.stat().st_size} bytes). Skipping download.")
        return True

    print(f"[Model Download] Downloading LaMa ONNX model (~208 MB) to {target_path}...")
    t0 = time.time()
    temp_path = target_path.with_suffix(".tmp")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(MODEL_URL, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as f:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 2 * 1024 * 1024

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = round(100.0 * downloaded / total_size, 1)
                    mb_done = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"LaMa Download Progress: {mb_done:.1f} MB / {mb_tot:.1f} MB ({pct}%)", end="\r")

        if temp_path.stat().st_size < MIN_BYTES:
            print(f"\n[Error] Downloaded file too small ({temp_path.stat().st_size} bytes).")
            if temp_path.exists():
                temp_path.unlink()
            return False

        if target_path.exists():
            target_path.unlink()
        temp_path.replace(target_path)
        print(f"\n[Model Download] Download finished successfully in {time.time() - t0:.1f}s! Final size: {target_path.stat().st_size} bytes.")
        return True
    except Exception as e:
        print(f"\n[Model Download] Failed to download model: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return False

if __name__ == "__main__":
    success = download_model()
    if not success and not (MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_BYTES):
        sys.exit(1)
