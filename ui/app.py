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

# API Endpoint Target
API_BASE_URL = f"http://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}"
PROCESS_ENDPOINT = f"{API_BASE_URL}/process"
# In local container/localhost setup, fallback to 127.0.0.1 if 0.0.0.0 is used
LOCAL_PROCESS_ENDPOINT = f"http://127.0.0.1:{settings.PORT}{settings.API_V1_STR}/process"


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


def build_ui() -> gr.Blocks:
    """
    Builds the enterprise-grade dual-pane QA interface using Gradio Blocks.
    """
    with gr.Blocks(title="WatermarkRemoverAI - Enterprise QA") as demo:
        with gr.Row(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 🖼️ WatermarkRemoverAI · Internal QA Console
                **Enterprise-grade Watermark Detection & LaMa Inpainting Platform**
                """
            )

        with gr.Row():
            # ==================================================================
            # LEFT PANE: Input Image with Interactive Brush Mask Tool
            # ==================================================================
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
                    "✨ Remove Watermark",
                    variant="primary",
                    size="lg"
                )

            # ==================================================================
            # RIGHT PANE: Cleaned Image & Diagnostic Verification
            # ==================================================================
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
                    value="*Awaiting execution... Upload an image and click 'Remove Watermark'.*",
                    elem_classes=["status-box"]
                )

        # Wire click event to async handler
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

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
        css=".gradio-container { max-width: 1350px !important; margin: 0 auto; } .header-box { text-align: center; padding: 20px 0 10px 0; } .status-box { border-radius: 8px; font-size: 0.95rem; }"
    )
