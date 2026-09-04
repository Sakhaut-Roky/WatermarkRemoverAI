"""
WatermarkRemoverAI - Enterprise SaaS Studio Interface
=====================================================
Dual-Studio Dark-Mode SaaS Platform for High-Precision Watermark Removal:
- Tab 1: "🖼️ Image Studio" - Single-image bounding-box / crop removal with multiple removal modes.
- Tab 2: "🎬 Video Studio" - Video keyframe bounding-box removal with audio preservation.

Both studios feature:
- Central Bounding-Box Canvas (gr.ImageEditor configured for box/crop selection without freehand drawing)
- "Box Mask Tool" Control Panel with coordinate tracking (X, Y, W, H)
- "Auto Detect" watermark scanner and "Clear" reset buttons
- Multi-algorithm Removal Modes:
    * Inpaint (Content-Aware Fill)
    * Gaussian Blur Blend
    * Pixelate
    * Smooth Edge Interpolation
- Vibrant Purple/Green Gradient Primary Action Buttons ("Cleanse Image" & "Cleanse Video")
"""

import os
import sys
import io
import json
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Ensure project root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from PIL import Image
import gradio as gr

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from core.config import settings

logger = logging.getLogger("WatermarkRemoverAI.UI")

# Backend API Endpoints
API_BASE_URL = f"http://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}"
PROCESS_ENDPOINT = f"{API_BASE_URL}/process"
PROCESS_VIDEO_ENDPOINT = f"{API_BASE_URL}/process_video"

# Localhost Fallbacks
LOCAL_PROCESS_ENDPOINT = f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}/process"
LOCAL_PROCESS_VIDEO_ENDPOINT = f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}/process_video"

# ==============================================================================
# Custom Sleek Dark-Mode SaaS CSS
# ==============================================================================
CUSTOM_CSS = """
/* === Root Variables & Global Theme === */
:root {
  --bg-main: #0a0b10;
  --bg-card: #121520;
  --bg-card-secondary: #171b2a;
  --bg-input: #0b0d15;
  --border-card: rgba(255, 255, 255, 0.08);
  --border-focus: #8b5cf6;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --accent-purple: #8b5cf6;
  --accent-emerald: #10b981;
}

body, .gradio-container {
  background-color: var(--bg-main) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  max-width: 1440px !important;
  margin: 0 auto !important;
}

/* === Modern SaaS Header === */
.saas-header {
  text-align: center;
  padding: 24px 0 16px 0;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border-card);
}
.brand-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(139, 92, 246, 0.12);
  border: 1px solid rgba(139, 92, 246, 0.35);
  color: #c084fc;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 4px 14px;
  border-radius: 9999px;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.saas-title {
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #ffffff;
  margin: 0;
}
.saas-subtitle {
  font-size: 0.95rem;
  color: var(--text-secondary);
  margin-top: 6px;
}

/* === Modern UI Cards === */
.saas-card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: 16px !important;
  padding: 20px !important;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
  backdrop-filter: blur(12px) !important;
  margin-bottom: 16px !important;
}

/* === Central Canvas Card === */
.central-canvas-card {
  background: #0d0f18 !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* === Box Mask Tool Control Panel === */
.box-mask-panel {
  background: var(--bg-card) !important;
  border: 1px solid rgba(139, 92, 246, 0.25) !important;
  border-radius: 16px !important;
  padding: 22px !important;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5) !important;
  margin-top: 14px !important;
}
.panel-header {
  margin-bottom: 16px;
}
.panel-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.02em;
  margin: 0 0 4px 0;
}
.panel-subtitle {
  font-size: 0.88rem;
  color: var(--text-secondary);
  margin: 0;
}

/* === Small Action Buttons === */
.btn-auto-detect {
  background: rgba(139, 92, 246, 0.16) !important;
  border: 1px solid rgba(139, 92, 246, 0.45) !important;
  color: #c084fc !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
}
.btn-auto-detect:hover {
  background: rgba(139, 92, 246, 0.28) !important;
  border-color: rgba(139, 92, 246, 0.75) !important;
  color: #ffffff !important;
  transform: translateY(-1px) !important;
}
.btn-clear {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(255, 255, 255, 0.12) !important;
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
}
.btn-clear:hover {
  background: rgba(239, 68, 68, 0.15) !important;
  border-color: rgba(239, 68, 68, 0.45) !important;
  color: #f87171 !important;
  transform: translateY(-1px) !important;
}

/* === Coordinate Inputs Styling === */
.coord-input input {
  background: var(--bg-input) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: #38bdf8 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.95rem !important;
  border-radius: 8px !important;
}

/* === Large Primary "Cleanse" Buttons (Purple / Green Gradient) === */
.cleanse-btn {
  background: linear-gradient(135deg, #8b5cf6 0%, #10b981 100%) !important;
  color: #ffffff !important;
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 16px 28px !important;
  box-shadow: 0 4px 20px rgba(139, 92, 246, 0.45), 0 0 35px rgba(16, 185, 129, 0.25) !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  cursor: pointer !important;
  text-transform: uppercase !important;
  margin-top: 14px !important;
  width: 100% !important;
}
.cleanse-btn:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 8px 30px rgba(139, 92, 246, 0.65), 0 0 50px rgba(16, 185, 129, 0.4) !important;
  filter: brightness(1.08) !important;
}
.cleanse-btn:active {
  transform: translateY(0) !important;
}

/* === Status / Output Card === */
.status-card {
  background: #0e111a !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  font-size: 0.9rem !important;
  color: #cbd5e1 !important;
  margin-top: 12px !important;
}
"""

# ==============================================================================
# Helper Utilities
# ==============================================================================

def extract_video_path(video_input: Any) -> Optional[str]:
    """Extracts raw filesystem string path from Gradio Video component output."""
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if hasattr(video_input, "name"):
        return str(video_input.name)
    if isinstance(video_input, dict):
        return video_input.get("video") or video_input.get("name") or video_input.get("path")
    return None


def extract_keyframe_from_video(video_path: str, timestamp_sec: float = 0.0) -> Optional[Image.Image]:
    """Extracts an RGB keyframe from video at specified timestamp."""
    if not video_path or not Path(video_path).is_file():
        return None
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(video_path)
        frame_t = min(max(0.0, timestamp_sec), max(0.0, (clip.duration or 1.0) - 0.05))
        frame_rgb = clip.get_frame(frame_t)
        clip.close()
        return Image.fromarray(frame_rgb)
    except Exception:
        # Fallback to OpenCV VideoCapture
        cap = cv2.VideoCapture(video_path)
        ret, frame_bgr = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
    return None


def extract_bbox_from_editor(editor_data: Any) -> Tuple[int, int, int, int]:
    """
    Extracts bounding box coordinates (x, y, width, height) from Gradio ImageEditor.
    When a user crops in ImageEditor:
      - 'background' is the uncropped original keyframe/image
      - 'composite' is the user's cropped slice
    Uses normalized cross-correlation (cv2.matchTemplate) to determine exact pixel offsets.
    Returns:
        (x, y, width, height) in pixel coordinates, or (0, 0, 0, 0) if no crop was drawn.
    """
    if not editor_data or not isinstance(editor_data, dict):
        return (0, 0, 0, 0)

    bg = editor_data.get("background")
    comp = editor_data.get("composite")
    if bg is None or comp is None:
        return (0, 0, 0, 0)

    try:
        bg_rgb = bg.convert("RGB") if isinstance(bg, Image.Image) else Image.fromarray(bg).convert("RGB")
        comp_rgb = comp.convert("RGB") if isinstance(comp, Image.Image) else Image.fromarray(comp).convert("RGB")

        bg_arr = np.array(bg_rgb)
        comp_arr = np.array(comp_rgb)

        # If dimensions are equal, user has not cropped
        if bg_arr.shape[:2] == comp_arr.shape[:2]:
            return (0, 0, 0, 0)

        bg_h, bg_w = bg_arr.shape[:2]
        comp_h, comp_w = comp_arr.shape[:2]

        if comp_h > bg_h or comp_w > bg_w or comp_h <= 0 or comp_w <= 0:
            return (0, 0, 0, 0)

        bg_gray = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2GRAY)
        comp_gray = cv2.cvtColor(comp_arr, cv2.COLOR_RGB2GRAY)

        res = cv2.matchTemplate(bg_gray, comp_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        x = int(max_loc[0])
        y = int(max_loc[1])
        w = int(comp_w)
        h = int(comp_h)
        return (x, y, w, h)
    except Exception as exc:
        logger.debug("Error computing bbox from crop: %s", exc)
        return (0, 0, 0, 0)


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Converts PIL Image to base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def base64_to_pil(b64_str: str) -> Optional[Image.Image]:
    """Decodes base64 data URI to PIL Image."""
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    decoded = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(decoded))


# ==============================================================================
# TAB 1: IMAGE STUDIO CALLBACKS
# ==============================================================================

def on_img_editor_changed(
    editor_data: Any,
    cur_x: int,
    cur_y: int,
    cur_w: int,
    cur_h: int
) -> Tuple[int, int, int, int, str]:
    """Synchronizes crop changes made in the Image Studio canvas into coordinate inputs."""
    x, y, w, h = extract_bbox_from_editor(editor_data)
    if w > 0 and h > 0:
        return x, y, w, h, f"📍 **Box Selected**: X: `{x}` | Y: `{y}` | Width: `{w}` | Height: `{h}`"
    return cur_x, cur_y, cur_w, cur_h, "*Draw a box tightly over the watermark to remove it completely.*"


def on_img_auto_detect_click(editor_data: Any) -> Tuple[int, int, int, int, Optional[Image.Image], str]:
    """Autonomous watermark detector for Image Studio."""
    img = None
    if editor_data and isinstance(editor_data, dict):
        img = editor_data.get("background") or editor_data.get("composite")
    elif isinstance(editor_data, Image.Image):
        img = editor_data

    if img is None:
        return 0, 0, 0, 0, None, "⚠️ **No image uploaded.** Please upload an image first."

    img_rgb = np.array(img.convert("RGB"))
    h_img, w_img = img_rgb.shape[:2]

    try:
        from api.router import get_service
        service = get_service()
        mask = service.detector.generate_mask(img_rgb)
    except Exception:
        from core.mask_generator import WatermarkDetector
        detector = WatermarkDetector(auto_fallback=True)
        mask = detector.generate_mask(img_rgb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        pad = 4
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2 * pad)
        h = min(h_img - y, h + 2 * pad)

        annotated = img_rgb.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (16, 185, 129), 3)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (139, 92, 246), -1)
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

        preview_pil = Image.fromarray(annotated)
        msg = f"⚡ **Auto-Detected Watermark**: X: `{x}` | Y: `{y}` | Width: `{w}` | Height: `{h}`"
        return x, y, w, h, preview_pil, msg
    else:
        return 0, 0, 0, 0, img, "ℹ️ **No watermark detected autonomously.** Please drag a box manually."


def on_img_clear_click(editor_data: Any) -> Tuple[int, int, int, int, Optional[Image.Image], str]:
    """Clears bounding box coordinates and resets image canvas."""
    clean_img = None
    if editor_data and isinstance(editor_data, dict):
        clean_img = editor_data.get("background")
    elif isinstance(editor_data, Image.Image):
        clean_img = editor_data
    return 0, 0, 0, 0, clean_img, "🧹 **Box mask cleared.** Draw a box tightly over the watermark."


async def on_cleanse_image_click(
    editor_data: Any,
    x_val: float,
    y_val: float,
    w_val: float,
    h_val: float,
    removal_mode: str,
    composite_flag: bool
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """
    Sends the image and bounding box coordinates [x, y, width, height]
    to the FastAPI /api/v1/process endpoint.
    """
    source_img = None
    if editor_data and isinstance(editor_data, dict):
        source_img = editor_data.get("background") or editor_data.get("composite")
    elif isinstance(editor_data, Image.Image):
        source_img = editor_data

    if source_img is None:
        return None, None, "⚠️ **No image uploaded.** Please upload an image in the left panel to proceed."

    x = int(x_val or 0)
    y = int(y_val or 0)
    w = int(w_val or 0)
    h = int(h_val or 0)

    if w <= 0 or h <= 0:
        bx, by, bw, bh = extract_bbox_from_editor(editor_data)
        if bw > 0 and bh > 0:
            x, y, w, h = bx, by, bw, bh

    bbox_list = [x, y, w, h] if (w > 0 and h > 0) else None
    img_b64 = pil_to_base64(source_img.convert("RGB"))

    payload = {
        "image_base64": img_b64,
        "bbox": bbox_list,
        "removal_mode": removal_mode,
        "composite": bool(composite_flag),
        "auto_fallback": True
    }

    target_url = LOCAL_PROCESS_ENDPOINT if "0.0.0.0" in settings.HOST else PROCESS_ENDPOINT

    if HAS_HTTPX:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(target_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    cleaned_img = base64_to_pil(data.get("result_base64"))
                    mask_img = base64_to_pil(data.get("mask_base64"))
                    dur = data.get("duration_seconds", 0.0)
                    jid = data.get("job_id", "N/A")
                    bbox_str = f"`[X: {x}, Y: {y}, W: {w}, H: {h}]`" if bbox_list else "Autonomous Detection"
                    msg = (
                        f"✅ **Image Cleanse Completed Successfully**\n\n"
                        f"- **Removal Mode**: `{removal_mode}`\n"
                        f"- **Box Mask**: {bbox_str}\n"
                        f"- **Processing Time**: `{dur}s`\n"
                        f"- **Job UUID**: `{jid}`"
                    )
                    return cleaned_img, mask_img, msg
                else:
                    return None, None, f"❌ **API Error ({response.status_code})**: {response.text}"
        except httpx.ConnectError:
            logger.warning("Could not reach API via HTTP. Executing in-process fallback.")
            from api.router import execute_watermark_removal, ProcessRequest
            import uuid
            req = ProcessRequest(**payload)
            jid = str(uuid.uuid4())
            data = execute_watermark_removal(jid, req)
            cleaned_img = base64_to_pil(data.get("result_base64"))
            mask_img = base64_to_pil(data.get("mask_base64"))
            dur = data.get("duration_seconds", 0.0)
            msg = (
                f"✅ **Image Cleanse Completed (In-Process Fallback)**\n\n"
                f"- **Removal Mode**: `{removal_mode}`\n"
                f"- **Processing Time**: `{dur}s`\n"
                f"- **Job UUID**: `{jid}`"
            )
            return cleaned_img, mask_img, msg
        except Exception as exc:
            return None, None, f"❌ **Request Error**: {str(exc)}"
    else:
        return None, None, "❌ **Dependency Error**: `httpx` is required for asynchronous API requests."


# ==============================================================================
# TAB 2: VIDEO STUDIO CALLBACKS
# ==============================================================================

def on_video_changed(video_file: Any) -> Tuple[Optional[Image.Image], int, int, int, int, str]:
    """
    Triggered when an MP4 video is uploaded.
    Extracts the first keyframe and loads it into the central ImageEditor for box drafting.
    """
    video_path = extract_video_path(video_file)
    if not video_path or not Path(video_path).is_file():
        return None, 0, 0, 0, 0, "*Upload an MP4 video to load keyframe canvas.*"

    frame_pil = extract_keyframe_from_video(video_path, timestamp_sec=0.0)
    if frame_pil is not None:
        return (
            frame_pil,
            0,
            0,
            0,
            0,
            f"🎬 **Loaded Keyframe** from `{Path(video_path).name}` ({frame_pil.width}x{frame_pil.height}px). "
            "Draw a box tightly over the watermark using the Crop tool below."
        )
    return None, 0, 0, 0, 0, "⚠️ Could not extract keyframe from uploaded video."


def on_video_editor_changed(
    editor_data: Any,
    cur_x: int,
    cur_y: int,
    cur_w: int,
    cur_h: int
) -> Tuple[int, int, int, int, str]:
    """Synchronizes crop changes made in the Video Studio canvas into coordinate inputs."""
    x, y, w, h = extract_bbox_from_editor(editor_data)
    if w > 0 and h > 0:
        msg = f"📍 **Box Selected**: X: `{x}` | Y: `{y}` | Width: `{w}` | Height: `{h}`"
        return x, y, w, h, msg
    return cur_x, cur_y, cur_w, cur_h, "*Draw a box tightly over the watermark to remove it completely.*"


def on_video_auto_detect_click(
    video_file: Any,
    editor_data: Any
) -> Tuple[int, int, int, int, Optional[Image.Image], str]:
    """Runs autonomous watermark detection (EasyOCR + SAM 2 / OpenCV) on the keyframe."""
    img = None
    if editor_data and isinstance(editor_data, dict):
        img = editor_data.get("background") or editor_data.get("composite")
    elif isinstance(editor_data, Image.Image):
        img = editor_data

    if img is None and video_file:
        vpath = extract_video_path(video_file)
        if vpath:
            img = extract_keyframe_from_video(vpath, 0.0)

    if img is None:
        return 0, 0, 0, 0, None, "⚠️ **No media loaded.** Please upload an MP4 video or image first."

    img_rgb = np.array(img.convert("RGB"))
    h_img, w_img = img_rgb.shape[:2]

    try:
        from api.router import get_service
        service = get_service()
        mask = service.detector.generate_mask(img_rgb)
    except Exception:
        from core.mask_generator import WatermarkDetector
        detector = WatermarkDetector(auto_fallback=True)
        mask = detector.generate_mask(img_rgb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        pad = 4
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2 * pad)
        h = min(h_img - y, h + 2 * pad)

        annotated = img_rgb.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (16, 185, 129), 3)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (139, 92, 246), -1)
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

        preview_pil = Image.fromarray(annotated)
        msg = f"⚡ **Auto-Detected Watermark**: X: `{x}` | Y: `{y}` | Width: `{w}` | Height: `{h}`"
        return x, y, w, h, preview_pil, msg
    else:
        return 0, 0, 0, 0, img, "ℹ️ **No watermark detected autonomously.** Please drag a box manually."


def on_video_clear_click(video_file: Any) -> Tuple[int, int, int, int, Optional[Image.Image], str]:
    """Clears all bounding box coordinates and reloads the clean keyframe."""
    clean_img = None
    if video_file:
        vpath = extract_video_path(video_file)
        if vpath:
            clean_img = extract_keyframe_from_video(vpath, 0.0)
    return 0, 0, 0, 0, clean_img, "🧹 **Box mask cleared.** Draw a box tightly over the watermark."


async def on_cleanse_video_click(
    video_file: Any,
    editor_data: Any,
    x_val: float,
    y_val: float,
    w_val: float,
    h_val: float,
    removal_mode: str,
    composite_flag: bool,
    max_duration_sec: float
) -> Tuple[Optional[str], str]:
    """
    Sends the video along with captured bounding box coordinates (X, Y, W, H)
    and selected removal mode to the FastAPI /api/v1/process_video endpoint.
    """
    video_path = extract_video_path(video_file)
    if not video_path or not Path(video_path).is_file():
        return None, "⚠️ **No video uploaded.** Please upload an MP4 video file to proceed."

    source_path = Path(video_path)
    if not source_path.name.lower().endswith(".mp4"):
        return None, f"❌ **Unsupported format**: '{source_path.name}'. Only `.mp4` video files are supported."

    x = int(x_val or 0)
    y = int(y_val or 0)
    w = int(w_val or 0)
    h = int(h_val or 0)

    if w <= 0 or h <= 0:
        bx, by, bw, bh = extract_bbox_from_editor(editor_data)
        if bw > 0 and bh > 0:
            x, y, w, h = bx, by, bw, bh

    bbox_param = f"{x},{y},{w},{h}" if (w > 0 and h > 0) else None
    target_url = LOCAL_PROCESS_VIDEO_ENDPOINT if "0.0.0.0" in settings.HOST else PROCESS_VIDEO_ENDPOINT

    logger.info(
        "Initiating Cleanse Video: File=%s | BBox=%s | Mode=%s",
        source_path.name,
        bbox_param,
        removal_mode
    )

    if HAS_HTTPX:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(source_path, "rb") as vf:
                    files = {"file": (source_path.name, vf, "video/mp4")}
                    params = {
                        "removal_mode": str(removal_mode),
                        "composite": bool(composite_flag),
                        "max_duration": float(max_duration_sec),
                    }
                    if bbox_param:
                        params["bbox"] = bbox_param

                    response = await client.post(target_url, files=files, params=params)

                if response.status_code == 200:
                    temp_out = tempfile.NamedTemporaryFile(suffix="_cleaned.mp4", delete=False)
                    temp_out.write(response.content)
                    temp_out.close()

                    job_id = response.headers.get("x-job-id", "N/A")
                    elapsed = response.headers.get("x-processing-time-seconds", "N/A")
                    frames = response.headers.get("x-frames-processed", "N/A")
                    orig_dur = response.headers.get("x-original-duration", "N/A")
                    proc_dur = response.headers.get("x-processed-duration", "N/A")
                    audio_pres = response.headers.get("x-audio-preserved", "N/A")

                    bbox_desc = f"Applied `[X: {x}, Y: {y}, W: {w}, H: {h}]`" if bbox_param else "Autonomous Detection"

                    status_msg = (
                        f"✅ **Video Cleanse Completed Successfully**\n\n"
                        f"- **Removal Mode**: `{removal_mode}`\n"
                        f"- **Box Mask**: {bbox_desc}\n"
                        f"- **Processing Time**: `{elapsed}s`\n"
                        f"- **Frames Rendered**: `{frames}`\n"
                        f"- **Duration**: `{proc_dur}s` *(Source: `{orig_dur}s`)*\n"
                        f"- **Audio Track Preserved**: `{audio_pres}`\n"
                        f"- **Job UUID**: `{job_id}`"
                    )
                    return temp_out.name, status_msg
                else:
                    return None, f"❌ **API Error ({response.status_code})**: {response.text}"

        except httpx.ConnectError:
            logger.warning("Could not reach FastAPI server via HTTP. Executing in-process video fallback.")
            try:
                from core.video_processor import VideoWatermarkRemover
                remover = VideoWatermarkRemover(max_duration_seconds=max_duration_sec)
                temp_out = tempfile.NamedTemporaryFile(suffix="_cleaned.mp4", delete=False)
                temp_out.close()

                bbox_tuple = (x, y, w, h) if bbox_param else None
                result = remover.process_video(
                    video_path=source_path,
                    output_path=temp_out.name,
                    bbox=bbox_tuple,
                    removal_mode=removal_mode,
                    composite=composite_flag,
                )
                status_msg = (
                    f"✅ **Success (In-Process)**: Video watermark cleansed!\n\n"
                    f"- **Removal Mode**: `{removal_mode}`\n"
                    f"- **Processing Time**: `{result.get('elapsed_time_seconds')}s`\n"
                    f"- **Frames Rendered**: `{result.get('total_frames_processed')}`\n"
                    f"- **Audio Track Preserved**: `{result.get('audio_preserved')}`\n"
                    f"- **Average Speed**: `{result.get('average_processing_fps')} FPS`"
                )
                return result["output_video"], status_msg
            except Exception as exc:
                return None, f"❌ **In-Process Fallback Error**: {str(exc)}"
        except Exception as exc:
            return None, f"❌ **Request Error**: {str(exc)}"
    else:
        return None, "❌ **Dependency Error**: `httpx` is required for asynchronous API requests."


# ==============================================================================
# UI Builder
# ==============================================================================

def build_ui() -> gr.Blocks:
    """
    Builds the sleek, dark-mode SaaS Gradio Blocks interface featuring a dual-tab layout:
    Tab 1: 🖼️ Image Studio
    Tab 2: 🎬 Video Studio
    """
    with gr.Blocks(title="WatermarkRemoverAI Studio", theme=gr.themes.Monochrome()) as demo:
        # Injected Custom CSS for dark-mode SaaS styling
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # SaaS Brand Header
        with gr.Row(elem_classes=["saas-header"]):
            with gr.Column():
                gr.HTML(
                    """
                    <div style="text-align: center;">
                        <div class="brand-pill">🛡️ PRO ENTERPRISE · STUDIO EDITION</div>
                        <h1 class="saas-title">WatermarkRemover<span style="color: #8b5cf6;">AI</span></h1>
                        <p class="saas-subtitle">Precision Multi-Modal Neural Watermark Removal Platform</p>
                    </div>
                    """
                )

        with gr.Tabs(elem_classes=["main-tabs"]):
            # ==================================================================
            # TAB 1: IMAGE STUDIO
            # ==================================================================
            with gr.Tab("🖼️ Image Studio", id="tab_image_studio"):
                with gr.Row():
                    # Left Column: Image Canvas & Box Mask Tool
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["saas-card"]):
                            gr.Markdown("### 1. Source Image & Bounding-Box Selector")
                            img_editor = gr.ImageEditor(
                                label="Input Image Canvas (Crop tool: Drag a tight box over watermark)",
                                type="pil",
                                brush=False,
                                eraser=False,
                                layers=False,
                                transforms=["crop"],
                                sources=["upload", "clipboard"],
                                elem_classes=["central-canvas-card"]
                            )

                        # Control Panel: "Box Mask Tool"
                        with gr.Group(elem_classes=["box-mask-panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Box Mask Tool</h3>
                                    <p class="panel-subtitle">Draw a box tightly over the watermark to remove it completely.</p>
                                </div>
                                """
                            )

                            # Two Small Buttons: "Auto Detect" and "Clear"
                            with gr.Row():
                                img_auto_detect_btn = gr.Button(
                                    "⚡ Auto Detect",
                                    size="sm",
                                    elem_classes=["btn-auto-detect"],
                                    scale=1
                                )
                                img_clear_btn = gr.Button(
                                    "🗑️ Clear",
                                    size="sm",
                                    elem_classes=["btn-clear"],
                                    scale=1
                                )

                            # Coordinates Display & Fine-tuning Inputs
                            with gr.Row():
                                img_x_input = gr.Number(
                                    label="X (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )
                                img_y_input = gr.Number(
                                    label="Y (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )
                                img_w_input = gr.Number(
                                    label="Width (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )
                                img_h_input = gr.Number(
                                    label="Height (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )

                            # Removal Mode (Exact 4 options requested)
                            img_removal_mode = gr.Radio(
                                label="Removal Mode",
                                choices=[
                                    "Smooth Edge Interpolation",
                                    "Gaussian Blur Blend",
                                    "Pixelate",
                                    "Inpaint (Content-Aware Fill)"
                                ],
                                value="Inpaint (Content-Aware Fill)",
                                interactive=True,
                            )

                            with gr.Accordion("⚙️ Advanced Options", open=False):
                                img_composite_toggle = gr.Checkbox(
                                    value=True,
                                    label="High-Fidelity Composite (Preserve clean background pixels byte-for-byte)"
                                )

                            # Large Primary "Cleanse Image" Button (Purple/Green Gradient)
                            cleanse_image_btn = gr.Button(
                                "✨ Cleanse Image",
                                size="lg",
                                elem_classes=["cleanse-btn"]
                            )

                    # Right Column: Image Output & Mask Verification
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["saas-card"]):
                            gr.Markdown("### 2. Cleaned Image Verification")
                            with gr.Tabs():
                                with gr.TabItem("✨ Cleaned Output"):
                                    img_output = gr.Image(
                                        label="Cleaned Image Output",
                                        type="pil",
                                        interactive=False
                                    )
                                with gr.TabItem("🎭 Generated Box Mask"):
                                    img_mask_output = gr.Image(
                                        label="Binary Watermark Mask (White = Cleaned)",
                                        type="pil",
                                        interactive=False
                                    )

                            img_status_card = gr.Markdown(
                                value="*Awaiting image cleanse... Upload an image, drag a box over watermark, and click 'Cleanse Image'.*",
                                elem_classes=["status-card"]
                            )

                # --------------------------------------------------------------
                # Event Bindings for Image Studio
                # --------------------------------------------------------------
                img_editor.change(
                    fn=on_img_editor_changed,
                    inputs=[img_editor, img_x_input, img_y_input, img_w_input, img_h_input],
                    outputs=[img_x_input, img_y_input, img_w_input, img_h_input, img_status_card]
                )

                img_auto_detect_btn.click(
                    fn=on_img_auto_detect_click,
                    inputs=[img_editor],
                    outputs=[img_x_input, img_y_input, img_w_input, img_h_input, img_editor, img_status_card]
                )

                img_clear_btn.click(
                    fn=on_img_clear_click,
                    inputs=[img_editor],
                    outputs=[img_x_input, img_y_input, img_w_input, img_h_input, img_editor, img_status_card]
                )

                cleanse_image_btn.click(
                    fn=on_cleanse_image_click,
                    inputs=[
                        img_editor,
                        img_x_input,
                        img_y_input,
                        img_w_input,
                        img_h_input,
                        img_removal_mode,
                        img_composite_toggle
                    ],
                    outputs=[img_output, img_mask_output, img_status_card]
                )

            # ==================================================================
            # TAB 2: VIDEO STUDIO
            # ==================================================================
            with gr.Tab("🎬 Video Studio", id="tab_video_studio"):
                with gr.Row():
                    # Left Column: Upload & Central Keyframe Canvas
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["saas-card"]):
                            gr.Markdown("### 1. Source Video & Keyframe Bounding-Box Selector")
                            video_input = gr.Video(
                                label="Source Video (.mp4, up to 10s processed)",
                                sources=["upload"],
                                interactive=True,
                            )

                            # Central ImageEditor configured specifically for bounding-box / crop selection
                            video_editor = gr.ImageEditor(
                                label="Keyframe Canvas (Crop tool: Drag a tight box over watermark)",
                                type="pil",
                                brush=False,
                                eraser=False,
                                layers=False,
                                transforms=["crop"],
                                sources=["upload", "clipboard"],
                                elem_classes=["central-canvas-card"]
                            )

                        # Control Panel: "Box Mask Tool"
                        with gr.Group(elem_classes=["box-mask-panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Box Mask Tool</h3>
                                    <p class="panel-subtitle">Draw a box tightly over the watermark to remove it completely.</p>
                                </div>
                                """
                            )

                            # Two Small Buttons: "Auto Detect" and "Clear"
                            with gr.Row():
                                video_auto_detect_btn = gr.Button(
                                    "⚡ Auto Detect",
                                    size="sm",
                                    elem_classes=["btn-auto-detect"],
                                    scale=1
                                )
                                video_clear_btn = gr.Button(
                                    "🗑️ Clear",
                                    size="sm",
                                    elem_classes=["btn-clear"],
                                    scale=1
                                )

                            # Coordinates Display & Fine-tuning Inputs
                            with gr.Row():
                                video_x_input = gr.Number(
                                    label="X (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )
                                video_y_input = gr.Number(
                                    label="Y (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )
                                video_w_input = gr.Number(
                                    label="Width (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )
                                video_h_input = gr.Number(
                                    label="Height (px)",
                                    value=0,
                                    precision=0,
                                    minimum=0,
                                    elem_classes=["coord-input"]
                                )

                            # Removal Mode (Exact 4 options requested)
                            video_removal_mode = gr.Radio(
                                label="Removal Mode",
                                choices=[
                                    "Smooth Edge Interpolation",
                                    "Gaussian Blur Blend",
                                    "Pixelate",
                                    "Inpaint (Content-Aware Fill)"
                                ],
                                value="Inpaint (Content-Aware Fill)",
                                interactive=True,
                            )

                            # Advanced Settings Accordion
                            with gr.Accordion("⚙️ Engine Tuning", open=False):
                                video_composite_toggle = gr.Checkbox(
                                    value=True,
                                    label="High-Fidelity Composite (Preserve clean background pixels byte-for-byte)"
                                )
                                max_duration_slider = gr.Slider(
                                    minimum=1.0,
                                    maximum=10.0,
                                    value=10.0,
                                    step=0.5,
                                    label="Safety Duration Ceiling (Strict 10s limit)"
                                )

                            # Large, prominent primary button with purple/green gradient style
                            cleanse_video_btn = gr.Button(
                                "✨ Cleanse Video",
                                size="lg",
                                elem_classes=["cleanse-btn"]
                            )

                    # Right Column: Output Video & Analytics
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["saas-card"]):
                            gr.Markdown("### 2. Cleaned Video Verification")
                            video_output = gr.Video(
                                label="Cleaned Video (Synchronized Original Audio)",
                                interactive=False,
                                autoplay=False
                            )

                            video_status_card = gr.Markdown(
                                value="*Awaiting video cleanse... Upload an MP4 video, drag a box over watermark, and click 'Cleanse Video'.*",
                                elem_classes=["status-card"]
                            )

                # --------------------------------------------------------------
                # Event Bindings for Video Studio
                # --------------------------------------------------------------
                # When video is uploaded, extract keyframe 0 into Video ImageEditor
                video_input.change(
                    fn=on_video_changed,
                    inputs=[video_input],
                    outputs=[video_editor, video_x_input, video_y_input, video_w_input, video_h_input, video_status_card]
                )

                # When user crops in Video ImageEditor, synchronize coordinates
                video_editor.change(
                    fn=on_video_editor_changed,
                    inputs=[video_editor, video_x_input, video_y_input, video_w_input, video_h_input],
                    outputs=[video_x_input, video_y_input, video_w_input, video_h_input, video_status_card]
                )

                # When user clicks "Auto Detect"
                video_auto_detect_btn.click(
                    fn=on_video_auto_detect_click,
                    inputs=[video_input, video_editor],
                    outputs=[video_x_input, video_y_input, video_w_input, video_h_input, video_editor, video_status_card]
                )

                # When user clicks "Clear"
                video_clear_btn.click(
                    fn=on_video_clear_click,
                    inputs=[video_input],
                    outputs=[video_x_input, video_y_input, video_w_input, video_h_input, video_editor, video_status_card]
                )

                # When user clicks "Cleanse Video"
                cleanse_video_btn.click(
                    fn=on_cleanse_video_click,
                    inputs=[
                        video_input,
                        video_editor,
                        video_x_input,
                        video_y_input,
                        video_w_input,
                        video_h_input,
                        video_removal_mode,
                        video_composite_toggle,
                        max_duration_slider
                    ],
                    outputs=[video_output, video_status_card]
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Monochrome(),
        css=CUSTOM_CSS,
    )
