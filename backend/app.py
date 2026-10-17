import os
import sys
import uuid
import time
import zipfile
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from backend.engine.detector import WatermarkDetector
from backend.engine.cleaner import WatermarkCleaner

detector = WatermarkDetector()
cleaner = WatermarkCleaner()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LogoErase API is ready. LaMa model will load on first cleanup.")
    yield

app = FastAPI(title="LogoErase Pro API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)
batches: Dict[str, Dict[str, Any]] = {}
MAX_PARALLEL_WORKERS = int(os.environ.get("MAX_PARALLEL_WORKERS", 10))

def clamp_parallel_workers(value: int) -> int:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = 1
    return max(1, min(MAX_PARALLEL_WORKERS, workers))

def cleanup_old_sessions():
    now = time.time()
    for item in SESSIONS_DIR.iterdir():
        if item.is_dir():
            try:
                mtime = item.stat().st_mtime
                if now - mtime > 7200:
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                pass

def process_single_image(
    batch_id: str,
    img_data: Dict[str, Any],
    mode: str = "lama",
    sensitivity: float = 1.0,
    feather: int = 2,
    custom_bbox: Optional[List[int]] = None,
    custom_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    orig_path = img_data["orig_path"]
    proc_path = img_data["proc_path"]
    mask_path = img_data["mask_path"]
    
    im = cv2.imread(orig_path)
    if im is None:
        img_data["status"] = "error"
        img_data["error"] = "Failed to decode image file"
        return img_data
        
    h_orig, w_orig = im.shape[:2]
    img_data["dimensions"] = [int(w_orig), int(h_orig)]
    
    # 1. Detection
    if custom_bbox is not None:
        bbox = custom_bbox
        confidence = 1.0
        flagged = False
        detected = True
        is_present = True
        msg = "User-specified region"
    else:
        det_res = detector.detect(im, sensitivity=sensitivity)
        bbox = det_res.bbox
        confidence = det_res.confidence
        flagged = det_res.flagged
        detected = det_res.detected
        is_present = det_res.is_present
        msg = det_res.message
        
    img_data["bbox"] = [int(v) for v in bbox]
    img_data["confidence"] = float(confidence)
    img_data["flagged"] = bool(flagged)
    img_data["detected"] = bool(detected)
    img_data["is_present"] = bool(is_present)
    img_data["message"] = str(msg)
    
    # 2. Cleaning
    try:
        if not is_present:
            # Safely preserve watermark-free image 100% untouched
            shutil.copy2(orig_path, proc_path)
            cv2.imwrite(mask_path, np.zeros((h_orig, w_orig), dtype=np.uint8))
            img_data["status"] = "completed"
            img_data["metadata"] = {"mode": mode, "modified_pixels": 0, "status": "untouched"}
            return img_data

        out_img, full_mask, meta = cleaner.clean(
            image=im,
            bbox=bbox,
            mode=mode,
            custom_mask=custom_mask,
            feather=feather,
            is_watermark_present=is_present,
            aggressive=bool(flagged or not detected)
        )
        
        ext = Path(orig_path).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            cv2.imwrite(proc_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        elif ext == ".png":
            cv2.imwrite(proc_path, out_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif ext == ".webp":
            cv2.imwrite(proc_path, out_img, [cv2.IMWRITE_WEBP_QUALITY, 98])
        else:
            cv2.imwrite(proc_path, out_img)
            
        cv2.imwrite(mask_path, full_mask)
        img_data["status"] = "completed"
        img_data["metadata"] = meta
    except Exception as e:
        img_data["status"] = "error"
        img_data["error"] = str(e)
        
    return img_data

def run_batch_job(batch_id: str, mode: str, sensitivity: float, feather: int, parallel_workers: int = 1):
    batch = batches.get(batch_id)
    if not batch:
        return
        
    batch["status"] = "processing"
    batch["start_time"] = time.time()
    batch["parallel_workers"] = clamp_parallel_workers(parallel_workers)
    images = batch["images"]
    total = len(images)
    
    completed_count = 0
    flagged_count = 0
    error_count = 0

    image_items = list(images.items())

    def process_worker(idx: int, img_id: str, img_data: Dict[str, Any]):
        if batch.get("canceled", False):
            img_data["status"] = "pending"
            return idx, img_id, img_data

        img_data["status"] = "processing"
        return idx, img_id, process_single_image(
            batch_id,
            img_data,
            mode=mode,
            sensitivity=sensitivity,
            feather=feather
        )

    with ThreadPoolExecutor(max_workers=batch["parallel_workers"]) as pool:
        futures = {}
        for idx, (img_id, img_data) in enumerate(image_items):
            if batch.get("canceled", False):
                batch["status"] = "canceled"
                break

            futures[pool.submit(process_worker, idx, img_id, img_data)] = (idx, img_id, img_data)

        for future in as_completed(futures):
            if batch.get("canceled", False):
                batch["status"] = "canceled"
                break

            idx, img_id, img_data = futures[future]
            try:
                _, _, img_data = future.result()
            except Exception as e:
                img_data["status"] = "error"
                img_data["error"] = str(e)

            if img_data["status"] == "completed":
                completed_count += 1
                if img_data.get("flagged", False):
                    flagged_count += 1
            elif img_data["status"] == "error":
                error_count += 1

            processed_count = completed_count + error_count
            batch["completed_count"] = completed_count
            batch["flagged_count"] = flagged_count
            batch["error_count"] = error_count
            batch["current_index"] = processed_count
            batch["current_filename"] = img_data.get("filename", "")
            batch["progress"] = round(100.0 * processed_count / total, 1)

            elapsed = max(0.01, time.time() - batch["start_time"])
            batch["elapsed_sec"] = round(elapsed, 1)
            batch["speed"] = round(processed_count / elapsed, 1)

    if batch.get("canceled", False):
        batch["status"] = "canceled"
    else:
        batch["status"] = "completed"
    batch["end_time"] = time.time()

    elapsed = max(0.01, batch["end_time"] - batch["start_time"])
    processed_count = completed_count + error_count
    batch["progress"] = 100.0 if total else 0.0
    batch["elapsed_sec"] = round(elapsed, 1)
    batch["speed"] = round(processed_count / elapsed, 1)

# ---------------- API ROUTES ----------------

@app.post("/api/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    paths: Optional[str] = Form(None)
):
    cleanup_old_sessions()
    batch_id = str(uuid.uuid4())[:8]
    session_path = SESSIONS_DIR / batch_id
    orig_dir = session_path / "original"
    proc_dir = session_path / "processed"
    mask_dir = session_path / "masks"
    
    orig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    rel_paths = paths.split(",") if paths else []
    batch_images = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    for i, file in enumerate(files):
        ext = Path(file.filename).suffix.lower()
        if ext not in valid_exts:
            continue
            
        img_id = f"img_{i:04d}"
        rel_path = rel_paths[i] if i < len(rel_paths) and rel_paths[i] else file.filename
        safe_name = f"{img_id}_{Path(file.filename).name}"
        save_path = orig_dir / safe_name
        
        with open(save_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
            
        proc_path = proc_dir / safe_name
        mask_path = mask_dir / f"{img_id}_mask.png"
        
        batch_images[img_id] = {
            "id": img_id,
            "filename": Path(file.filename).name,
            "rel_path": rel_path,
            "orig_path": str(save_path),
            "proc_path": str(proc_path),
            "mask_path": str(mask_path),
            "size_bytes": save_path.stat().st_size,
            "status": "pending",
            "confidence": None,
            "flagged": False,
            "detected": None,
            "bbox": None
        }
        
    if not batch_images:
        raise HTTPException(status_code=400, detail="No supported image files provided.")
        
    batches[batch_id] = {
        "batch_id": batch_id,
        "status": "ready",
        "created_at": time.time(),
        "total_count": len(batch_images),
        "completed_count": 0,
        "flagged_count": 0,
        "error_count": 0,
        "progress": 0.0,
        "speed": 0.0,
        "elapsed_sec": 0.0,
        "parallel_workers": 1,
        "images": batch_images
    }
    
    return {"batch_id": batch_id, "total": len(batch_images)}

class ProcessRequest(BaseModel):
    batch_id: str
    mode: str = "lama"
    sensitivity: float = 1.0
    feather: int = 2
    parallel_workers: int = 1

@app.post("/api/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    batch = batches.get(req.batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    if batch["status"] == "processing":
        return {"status": "already_running"}

    batch.update(status="processing", completed_count=0, flagged_count=0,
                 error_count=0, progress=0.0, current_index=0,
                 parallel_workers=clamp_parallel_workers(req.parallel_workers))
    for image in batch["images"].values():
        image["status"] = "pending"
        image.pop("error", None)
        
    background_tasks.add_task(
        run_batch_job,
        batch_id=req.batch_id,
        mode=req.mode,
        sensitivity=req.sensitivity,
        feather=req.feather,
        parallel_workers=req.parallel_workers
    )
    
    return {"status": "started", "batch_id": req.batch_id}

@app.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch

@app.get("/api/image/{batch_id}/{img_id}")
async def get_image(batch_id: str, img_id: str, type: str = "processed"):
    batch = batches.get(batch_id)
    if not batch or img_id not in batch["images"]:
        raise HTTPException(status_code=404, detail="Image not found")
        
    img_data = batch["images"][img_id]
    if type == "original":
        path = img_data["orig_path"]
    elif type == "mask":
        path = img_data["mask_path"]
    else:
        path = img_data["proc_path"] if Path(img_data["proc_path"]).exists() else img_data["orig_path"]
        
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not ready")
        
    return FileResponse(path)

class ReprocessRequest(BaseModel):
    batch_id: str
    img_id: str
    mode: str = "lama"
    bbox: Optional[List[int]] = None
    feather: int = 2

@app.post("/api/reprocess")
def reprocess_image(req: ReprocessRequest):
    batch = batches.get(req.batch_id)
    if not batch or req.img_id not in batch["images"]:
        raise HTTPException(status_code=404, detail="Image not found")
        
    img_data = batch["images"][req.img_id]
    if batch["status"] == "processing":
        raise HTTPException(status_code=409, detail="Wait for the batch to finish before re-cleaning.")
    res = process_single_image(
        batch_id=req.batch_id,
        img_data=img_data,
        mode=req.mode,
        custom_bbox=req.bbox,
        feather=req.feather
    )
    if res["status"] == "error":
        raise HTTPException(status_code=500, detail=res.get("error", "Re-clean failed"))
    return res

@app.get("/api/download/{batch_id}/{img_id}")
async def download_image(batch_id: str, img_id: str):
    batch = batches.get(batch_id)
    if not batch or img_id not in batch["images"]:
        raise HTTPException(status_code=404, detail="Image not found")
        
    img_data = batch["images"][img_id]
    path = img_data["proc_path"] if Path(img_data["proc_path"]).exists() else img_data["orig_path"]
    return FileResponse(path, filename=img_data["filename"], media_type="application/octet-stream")

@app.get("/api/download-zip/{batch_id}")
def download_batch_zip(batch_id: str):
    batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    zip_path = SESSIONS_DIR / batch_id / f"cleaned_images_{batch_id}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_f:
        for img_id, data in batch["images"].items():
            p_path = Path(data["proc_path"])
            file_to_pack = p_path if p_path.exists() else Path(data["orig_path"])
            arcname = data.get("rel_path") or data["filename"]
            zip_f.write(file_to_pack, arcname=arcname)
            
    return FileResponse(
        zip_path,
        filename=f"logo_removed_images_{batch_id}.zip",
        media_type="application/zip"
    )

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "LogoErase", "max_parallel_workers": MAX_PARALLEL_WORKERS}


@app.middleware("http")
async def prevent_stale_app(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


STATIC_DIR = BASE_DIR / "frontend"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    uvicorn.run(app, host=host, port=port)

