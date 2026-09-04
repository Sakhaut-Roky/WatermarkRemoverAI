"""
WatermarkRemoverAI - Internal QA & Verification Interface
=========================================================
Dual-pane enterprise Gradio application featuring:
- Interactive brush tool for human-in-the-loop custom masking.
- Asynchronous communication with FastAPI /api/v1/process endpoint via base64 payloads.
- High-fidelity before/after comparison and intermediate mask inspection.
"""

import os
import sys
import io
import json
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Ensure project root directory is in sys.path for standalone executions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import gradio as gr

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from core.config import settings

logger = logging.getLogger("WatermarkRemoverAI.UI")

# API Endpoint Targets
API_BASE_URL = f"http://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}"
PROCESS_ENDPOINT = f"{API_BASE_URL}/process"
PROCESS_VIDEO_ENDPOINT = f"{API_BASE_URL}/process_video"

# In local container/localhost setup, fallback to 127.0.0.1 if 0.0.0.0 is used
LOCAL_PROCESS_ENDPOINT = f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}/process"
LOCAL_PROCESS_VIDEO_ENDPOINT = f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}/process_video"


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 encoded data URI."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def base64_to_pil(b64_str: str) -> Optional[Image.Image]:
    """Decodes a base64 data URI into a PIL Image."""
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    decoded_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(decoded_bytes))


def extract_mask_from_editor(editor_data: Any) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Extracts the source image and user-drawn brush mask from Gradio ImageEditor payload.
    
    Returns:
        (source_image, user_drawn_mask)
    """
    if editor_data is None:
        return None, None

    # Gradio 4.x/6.x ImageEditor returns a dictionary:
    # {'background': PIL.Image, 'layers': [PIL.Image, ...], 'composite': PIL.Image}
    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        layers = editor_data.get("layers", [])

        if bg is None:
            bg = editor_data.get("composite")

        if bg is None:
            return None, None

        # Convert to RGB if palette/RGBA
        if bg.mode != "RGB":
            bg = bg.convert("RGB")

        # Check if user drew on any canvas layer
        user_mask = None
        if layers:
            for layer in layers:
                if layer is not None:
                    # Look at alpha channel or RGB values of layer strokes
                    layer_rgba = layer.convert("RGBA")
                    arr = np.array(layer_rgba)
                    alpha = arr[:, :, 3]
                    rgb_max = arr[:, :, :3].max(axis=2)
                    has_drawing = np.any(alpha > 10) or np.any(rgb_max > 10)

                    if has_drawing:
                        binary_arr = np.where((alpha > 10) | (rgb_max > 10), 255, 0).astype(np.uint8)
                        user_mask = Image.fromarray(binary_arr, mode="L")
                        break

        return bg, user_mask

    elif isinstance(editor_data, Image.Image):
        return editor_data.convert("RGB"), None
    elif isinstance(editor_data, np.ndarray):
        return Image.fromarray(editor_data).convert("RGB"), None

    return None, None


async def remove_watermark_async(
    editor_data: Any,
    confidence_thresh: float,
    composite_flag: bool,
    fallback_flag: bool
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """
    Encodes image and user-drawn brush mask to base64, dispatches async POST request
    to FastAPI /api/v1/process, and renders the cleaned result.
    """
    source_img, user_mask = extract_mask_from_editor(editor_data)

    if source_img is None:
        return None, None, "⚠️ **No image uploaded.** Please upload an image in the left panel to proceed."

    # 1. Base64 Encoding
    img_b64 = pil_to_base64(source_img)
    mask_b64 = pil_to_base64(user_mask) if user_mask is not None else None

    payload = {
        "image_base64": img_b64,
        "mask_base64": mask_b64,
        "confidence_threshold": float(confidence_thresh),
        "composite": bool(composite_flag),
        "auto_fallback": bool(fallback_flag)
    }

    status_msg = ""
    mask_type = "Human-in-the-Loop Brush Mask" if mask_b64 else "Automated AI Detection"

    # 2. Async HTTP POST to FastAPI /api/v1/process
    if HAS_HTTPX:
        async with httpx.AsyncClient(timeout=120.0) as client:
            target_url = LOCAL_PROCESS_ENDPOINT if "0.0.0.0" in settings.HOST else PROCESS_ENDPOINT
            try:
                response = await client.post(target_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    cleaned_img = base64_to_pil(data.get("result_base64"))
                    mask_img = base64_to_pil(data.get("mask_base64"))
                    duration = data.get("duration_seconds", 0.0)

                    status_msg = (
                        f"✅ **Success**: Processed in `{duration}s`\n\n"
                        f"- **Mode**: `{mask_type}`\n"
                        f"- **Job UUID**: `{data.get('job_id')}`\n"
                        f"- **Status**: `{data.get('status')}`"
                    )
                    return cleaned_img, mask_img, status_msg
                else:
                    return None, None, f"❌ **API Error ({response.status_code})**: {response.text}"
            except httpx.ConnectError:
                logger.warning("Could not reach API server via HTTP. Invoking in-process inference engine.")
                # Direct in-process fallback when UI is launched standalone without uvicorn
                from api.router import execute_watermark_removal, ProcessRequest
                import uuid
                req = ProcessRequest(**payload)
                job_id = str(uuid.uuid4())
                data = execute_watermark_removal(job_id, req)
                cleaned_img = base64_to_pil(data.get("result_base64"))
                mask_img = base64_to_pil(data.get("mask_base64"))
                status_msg = (
                    f"✅ **Success (In-Process Fallback)**: Processed in `{data.get('duration_seconds')}s`\n\n"
                    f"- **Mode**: `{mask_type}`\n"
                    f"- **Job UUID**: `{job_id}`"
                )
                return cleaned_img, mask_img, status_msg
    else:
        return None, None, "❌ **Dependency Error**: `httpx` is required for asynchronous API requests."


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


async def remove_video_watermark_async(
    video_input: Any,
    static_mask: bool,
    composite: bool,
    max_duration: float
) -> Tuple[Optional[str], str]:
    """
    Asynchronously transmits the video file to the /api/v1/process_video endpoint,
    receives the cleaned MP4 stream, writes it to a persistent local cache for Gradio,
    and reports execution diagnostics.
    """
    video_path = extract_video_path(video_input)
    if not video_path or not Path(video_path).is_file():
        return None, "⚠️ **No video uploaded.** Please upload an MP4 video file to proceed."

    source_path = Path(video_path)
    if not source_path.name.lower().endswith(".mp4"):
        return None, f"❌ **Unsupported format**: '{source_path.name}'. Only `.mp4` video files are supported."

    target_url = LOCAL_PROCESS_VIDEO_ENDPOINT if "0.0.0.0" in settings.HOST else PROCESS_VIDEO_ENDPOINT

    if HAS_HTTPX:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(source_path, "rb") as vf:
                    files = {"file": (source_path.name, vf, "video/mp4")}
                    params = {
                        "static_mask": bool(static_mask),
                        "composite": bool(composite),
                        "max_duration": float(max_duration),
                    }
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

                    status_msg = (
                        f"✅ **Success: Video Watermark Removed**\n\n"
                        f"- **Processing Time**: `{elapsed}s`\n"
                        f"- **Frames Rendered**: `{frames}`\n"
                        f"- **Processed Duration**: `{proc_dur}s` *(Source: `{orig_dur}s`)*\n"
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
                remover = VideoWatermarkRemover(max_duration_seconds=max_duration)
                temp_out = tempfile.NamedTemporaryFile(suffix="_cleaned.mp4", delete=False)
                temp_out.close()
                result = remover.process_video(
                    video_path=source_path,
                    output_path=temp_out.name,
                    static_mask=static_mask,
                    composite=composite,
                )
                status_msg = (
                    f"✅ **Success (In-Process Fallback)**: Video watermark removed!\n\n"
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
        return None, "❌ **Dependency Error**: `httpx` is required for asynchronous video processing requests."


CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto; }
.header-box { text-align: center; padding: 20px 0 10px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 15px; }
.status-box { border-radius: 8px; font-size: 0.95rem; padding: 12px; }
"""


def build_ui() -> gr.Blocks:
    """
    Builds the enterprise-grade dual-mode QA interface (Image & Video Processing) using Gradio Blocks.
    """
    with gr.Blocks(title="WatermarkRemoverAI - Enterprise QA") as demo:
        with gr.Row(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🛡️ WatermarkRemoverAI · Enterprise QA & Processing Console
                **Multi-Modal Autonomous AI Platform for Image & Video Watermark Removal**
                """
            )

        with gr.Tabs(elem_classes=["main-tabs"]):
            # ==================================================================
            # TAB 1: IMAGE PROCESSING
            # ==================================================================
            with gr.Tab("🖼️ Image Processing", id="tab_image"):
                with gr.Row():
                    # Left Column: Image Input & Masking
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Source Image & Interactive Masking")
                        image_editor = gr.ImageEditor(
                            label="Input Image (Draw with brush to force a custom mask, or leave clean for auto-detection)",
                            type="pil",
                            brush=gr.Brush(
                                colors=["#FFFFFF", "#FF0000"],
                                default_size=25,
                                color_mode="fixed"
                            ),
                            eraser=gr.Eraser(default_size=25),
                            sources=["upload", "clipboard"],
                            transforms=["crop"]
                        )

                        with gr.Accordion("⚙️ Inference Controls", open=False):
                            confidence_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=settings.DEFAULT_CONFIDENCE_THRESHOLD,
                                step=0.05,
                                label="Detection Confidence Threshold"
                            )
                            composite_toggle = gr.Checkbox(
                                value=True,
                                label="Preserve Unmasked Background (High-Fidelity Alpha Composite)"
                            )
                            fallback_toggle = gr.Checkbox(
                                value=True,
                                label="Enable Classical OpenCV Fallback on GPU OOM"
                            )

                        remove_btn = gr.Button(
                            "✨ Remove Image Watermark",
                            variant="primary",
                            size="lg"
                        )

                    # Right Column: Image Output
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Inpainted Verification Output")
                        with gr.Tabs():
                            with gr.TabItem("✨ Cleaned Output"):
                                output_image = gr.Image(
                                    label="Inpainted Clean Image",
                                    type="pil",
                                    interactive=False
                                )
                            with gr.TabItem("🎭 Watermark Mask"):
                                output_mask = gr.Image(
                                    label="Binary Watermark Mask (White = Inpainted)",
                                    type="pil",
                                    interactive=False
                                )

                        status_output = gr.Markdown(
                            value="*Awaiting execution... Upload an image and click 'Remove Image Watermark'.*",
                            elem_classes=["status-box"]
                        )

                # Wire Image click event
                remove_btn.click(
                    fn=remove_watermark_async,
                    inputs=[
                        image_editor,
                        confidence_slider,
                        composite_toggle,
                        fallback_toggle
                    ],
                    outputs=[
                        output_image,
                        output_mask,
                        status_output
                    ]
                )

            # ==================================================================
            # TAB 2: VIDEO PROCESSING
            # ==================================================================
            with gr.Tab("🎬 Video Processing", id="tab_video"):
                with gr.Row():
                    # Left Column: Video Input & Pipeline Controls
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Source Video Upload")
                        video_input = gr.Video(
                            label="Input Video (.mp4, strictly up to 10s processed)",
                            sources=["upload"],
                            interactive=True
                        )

                        with gr.Accordion("⚙️ Video Pipeline Settings", open=True):
                            video_static_mask = gr.Checkbox(
                                value=True,
                                label="⚡ Fast Static Mode (Reuse detected mask across all frames)"
                            )
                            video_composite = gr.Checkbox(
                                value=True,
                                label="🎨 High-Fidelity Composite (Preserve unmasked background pixels)"
                            )
                            video_max_duration = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=10.0,
                                step=0.5,
                                label="⏱️ Max Processing Duration Ceiling (Seconds)"
                            )

                        video_remove_btn = gr.Button(
                            "🎬 Remove Video Watermark",
                            variant="primary",
                            size="lg"
                        )

                    # Right Column: Video Output
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Cleaned Video Output")
                        video_output = gr.Video(
                            label="Cleaned Video (with Synchronized Original Audio)",
                            interactive=False,
                            autoplay=False
                        )

                        video_status_output = gr.Markdown(
                            value="*Awaiting execution... Upload an MP4 video and click 'Remove Video Watermark'.*",
                            elem_classes=["status-box"]
                        )

                # Wire Video click event
                video_remove_btn.click(
                    fn=remove_video_watermark_async,
                    inputs=[
                        video_input,
                        video_static_mask,
                        video_composite,
                        video_max_duration
                    ],
                    outputs=[
                        video_output,
                        video_status_output
                    ]
                )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
