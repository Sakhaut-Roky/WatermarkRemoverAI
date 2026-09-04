"""
FastAPI RESTful Routing Module for WatermarkRemoverAI
====================================================
Provides asynchronous, high-concurrency endpoints:
- POST /api/v1/upload: Streams raw image into UUID-categorized job storage.
- POST /api/v1/process: Asynchronously coordinates WatermarkDetector and InpaintingLaMa.
  Supports both previously uploaded job_id and direct base64 image/mask payloads.
- GET  /api/v1/result/{job_id}: Streams cleaned image or returns processing metadata.
- GET  /api/v1/status/{job_id}: Returns job lifecycle state and diagnostics.
- GET  /api/v1/health: Service health probe.
"""

import os
import time
import uuid
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    import aiofiles
    import aiofiles.os
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from core.config import settings
from core.mask_generator import WatermarkDetector
from core.inpaint_engine import InpaintingLaMa

# Router instantiation
router = APIRouter(tags=["Watermark Removal Pipeline"])

# Storage configuration
STORAGE_ROOT = Path(settings.STORAGE_DIR)
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------

class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique UUID tracking the processing job")
    filename: str = Field(..., description="Original filename of uploaded asset")
    size_bytes: int = Field(..., description="Uploaded payload size in bytes")
    status: str = Field("uploaded", description="Current state of the job")
    created_at: str = Field(..., description="ISO timestamp of upload")


class ProcessRequest(BaseModel):
    job_id: Optional[str] = Field(None, description="UUID of previously uploaded image")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string for direct processing")
    mask_base64: Optional[str] = Field(None, description="Optional base64-encoded user-drawn brush mask")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for mask detection")
    composite: bool = Field(True, description="Enforce byte-for-byte fidelity in clean background areas")
    auto_fallback: bool = Field(True, description="Permit automatic CPU/OpenCV fallback on GPU OOM")


class ProcessResponse(BaseModel):
    job_id: str = Field(..., description="Unique UUID tracking the processing job")
    status: str = Field(..., description="Pipeline execution outcome (completed/processing/failed)")
    message: str = Field(..., description="Informational message regarding the execution")
    duration_seconds: float = Field(..., description="Elapsed processing wall time")
    result_url: str = Field(..., description="Endpoint to download or inspect cleaned image")
    mask_url: str = Field(..., description="Endpoint to inspect binary watermark mask")
    result_base64: Optional[str] = Field(None, description="Base64-encoded cleaned image for immediate UI rendering")
    mask_base64: Optional[str] = Field(None, description="Base64-encoded binary mask")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    original_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    diagnostics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ------------------------------------------------------------------------------
# Base64 Utilities
# ------------------------------------------------------------------------------

def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decodes base64 string to BGR image array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 image data")
    return img


def decode_base64_mask(b64_str: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Decodes base64 string to single-channel binary mask {0, 255}."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    nparr = np.frombuffer(data, np.uint8)
    # Could be RGBA from canvas or grayscale
    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Could not decode base64 mask data")

    if decoded.ndim == 3:
        if decoded.shape[2] == 4:
            # Extract alpha channel or non-zero drawn strokes
            alpha = decoded[:, :, 3]
            rgb_sum = decoded[:, :, :3].sum(axis=2)
            mask = np.where((alpha > 10) | (rgb_sum > 10), 255, 0).astype(np.uint8)
        else:
            gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            mask = np.where(gray > 20, 255, 0).astype(np.uint8)
    else:
        mask = np.where(decoded > 20, 255, 0).astype(np.uint8)

    if target_shape and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask


def encode_image_to_base64(img_bgr_or_rgb: np.ndarray, is_rgb: bool = True) -> str:
    """Encodes NumPy image to base64 PNG data URL."""
    img = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR) if is_rgb else img_bgr_or_rgb
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image to base64")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


# ------------------------------------------------------------------------------
# Pipeline Singleton Service
# ------------------------------------------------------------------------------

class InferenceService:
    """Thread-safe singleton holding references to detection and inpainting engines."""
    _instance: Optional["InferenceService"] = None

    def __init__(self):
        self.detector = WatermarkDetector(auto_fallback=True)
        self.inpainter = InpaintingLaMa()

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = InferenceService()
        return cls._instance


def get_service() -> InferenceService:
    return InferenceService.get_instance()


# ------------------------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------------------------

def get_job_dir(job_id: str) -> Path:
    """Returns directory path for specified job UUID and validates directory exists."""
    return STORAGE_ROOT / job_id


async def write_file_async(destination: Path, upload_file: UploadFile) -> int:
    """Writes an UploadFile stream asynchronously using aiofiles with fallback."""
    bytes_written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    if HAS_AIOFILES:
        async with aiofiles.open(destination, "wb") as buffer:
            while chunk := await upload_file.read(1024 * 1024):  # 1MB chunks
                await buffer.write(chunk)
                bytes_written += len(chunk)
    else:
        def _sync_write():
            nonlocal bytes_written
            with open(destination, "wb") as buffer:
                while True:
                    chunk = upload_file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    bytes_written += len(chunk)

        await asyncio.to_thread(_sync_write)

    return bytes_written


def read_metadata(job_dir: Path) -> Dict[str, Any]:
    """Reads job metadata JSON."""
    meta_path = job_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(job_dir: Path, data: Dict[str, Any]) -> None:
    """Writes job metadata JSON."""
    meta_path = job_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def execute_watermark_removal(job_id: str, request: ProcessRequest) -> Dict[str, Any]:
    """
    Synchronous CPU/GPU heavy pipeline worker executed inside threadpool.
    Supports both file-based jobs and direct base64 image/mask inputs.
    """
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(job_dir)
    service = get_service()
    start_time = time.time()

    # Determine input image source
    if request.image_base64:
        img_bgr = decode_base64_image(request.image_base64)
        original_path = job_dir / "original.png"
        cv2.imwrite(str(original_path), img_bgr)
        metadata["saved_filename"] = "original.png"
    else:
        original_path = job_dir / metadata.get("saved_filename", "original.png")
        if not original_path.is_file():
            raise FileNotFoundError(f"Original image asset missing for job: {job_id}")
        img_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)

    # Immediately convert BGR from cv2.imread/cv2.imdecode to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Determine mask source (Human-in-the-loop custom mask or automated detector)
    if request.mask_base64:
        mask = decode_base64_mask(request.mask_base64, target_shape=(h, w))
        diagnostics = {
            "method_used": "human_in_the_loop_brush",
            "coverage_percentage": float(np.count_nonzero(mask == 255) / mask.size * 100),
            "device": "user_canvas"
        }
    else:
        mask, diagnostics = service.detector.generate_mask(
            image=img_rgb,
            return_diagnostics=True
        )

    mask_path = job_dir / "mask.png"
    service.detector.save_mask(mask, mask_path)

    # Execute LaMa Inpainting using verified RGB image
    cleaned_rgb = service.inpainter.inpaint(
        image=img_rgb,
        mask=mask,
        composite=request.composite
    )
    result_path = job_dir / "cleaned.png"
    # save_result converts RGB to BGR before writing with cv2.imwrite to disk
    service.inpainter.save_result(cleaned_rgb, result_path)

    duration = round(time.time() - start_time, 4)

    # Encode base64 strings for direct client response (Gradio renders in RGB)
    result_base64 = encode_image_to_base64(cleaned_rgb, is_rgb=True)
    mask_base64 = encode_image_to_base64(mask, is_rgb=False)

    metadata.update({
        "job_id": job_id,
        "status": "completed",
        "duration_seconds": duration,
        "mask_file": str(mask_path.name),
        "result_file": str(result_path.name),
        "diagnostics": diagnostics,
        "completed_at": datetime.utcnow().isoformat()
    })
    write_metadata(job_dir, metadata)

    metadata["result_base64"] = result_base64
    metadata["mask_base64"] = mask_base64

    return metadata


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@router.get("/health", tags=["Health"], summary="Service health probe")
async def health_check():
    """Health check endpoint to verify service availability and hardware device."""
    service = get_service()
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "device": str(service.inpainter.device),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload image asset for watermark removal"
)
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts image file upload, allocates unique UUID workspace,
    and asynchronously writes content to disk via aiofiles.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type: '{file.content_type}'. Must be an image."
        )

    job_id = str(uuid.uuid4())
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    original_filename = file.filename or "upload.png"
    ext = Path(original_filename).suffix or ".png"
    target_filename = f"original{ext}"
    target_path = job_dir / target_filename

    bytes_written = await write_file_async(target_path, file)

    now_iso = datetime.utcnow().isoformat()
    metadata = {
        "job_id": job_id,
        "original_filename": original_filename,
        "saved_filename": target_filename,
        "size_bytes": bytes_written,
        "status": "uploaded",
        "created_at": now_iso
    }
    write_metadata(job_dir, metadata)

    return UploadResponse(
        job_id=job_id,
        filename=original_filename,
        size_bytes=bytes_written,
        status="uploaded",
        created_at=now_iso
    )


@router.post(
    "/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Asynchronously process watermark removal"
)
async def process_image(request: ProcessRequest):
    """
    Executes watermark detection and LaMa inpainting pipeline.
    Supports either job_id from /upload or direct base64 image and brush mask payloads.
    Runs computation inside worker thread to preserve event loop responsiveness.
    """
    if not request.job_id and not request.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'job_id' or 'image_base64'."
        )

    job_id = request.job_id or str(uuid.uuid4())
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(job_dir)
    metadata["status"] = "processing"
    write_metadata(job_dir, metadata)

    try:
        result_meta = await asyncio.to_thread(execute_watermark_removal, job_id, request)

        return ProcessResponse(
            job_id=job_id,
            status="completed",
            message="Watermark removed successfully",
            duration_seconds=result_meta.get("duration_seconds", 0.0),
            result_url=f"/api/v1/result/{job_id}",
            mask_url=f"/api/v1/result/{job_id}?artifact=mask",
            result_base64=result_meta.get("result_base64"),
            mask_base64=result_meta.get("mask_base64")
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        write_metadata(job_dir, metadata)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inpainting pipeline error: {str(exc)}"
        )


@router.get(
    "/result/{job_id}",
    summary="Retrieve cleaned result image or intermediate mask"
)
async def get_result(
    job_id: str,
    artifact: str = Query("cleaned", enum=["cleaned", "mask", "original"]),
    download: bool = Query(False, description="Force browser download")
):
    """Streams requested image artifact (cleaned image, binary mask, or original source)."""
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found."
        )

    metadata = read_metadata(job_dir)
    status_str = metadata.get("status")

    if status_str == "processing":
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"job_id": job_id, "status": "processing", "message": "Inpainting in progress."}
        )
    elif status_str == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job failed: {metadata.get('error', 'Unknown error')}"
        )

    if artifact == "mask":
        target_path = job_dir / metadata.get("mask_file", "mask.png")
    elif artifact == "original":
        target_path = job_dir / metadata.get("saved_filename", "original.png")
    else:
        target_path = job_dir / metadata.get("result_file", "cleaned.png")

    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact}' not found for job '{job_id}'."
        )

    filename = f"{artifact}_{job_id}.png"
    headers = {"Content-Disposition": f"attachment; filename={filename}"} if download else None

    return FileResponse(
        path=target_path,
        media_type="image/png",
        filename=filename if download else None,
        headers=headers
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Query job lifecycle status and diagnostics"
)
async def get_status(job_id: str):
    """Returns real-time status and diagnostics for a job."""
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found."
        )

    metadata = read_metadata(job_dir)
    return JobStatusResponse(
        job_id=job_id,
        status=metadata.get("status", "unknown"),
        created_at=metadata.get("created_at", ""),
        original_file=metadata.get("original_filename"),
        duration_seconds=metadata.get("duration_seconds"),
        diagnostics=metadata.get("diagnostics"),
        error=metadata.get("error")
    )
