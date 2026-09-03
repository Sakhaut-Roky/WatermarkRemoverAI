"""
WatermarkRemoverAI - Main Application Entry Point
------------------------------------------------
Enterprise-grade platform orchestrating REST APIs, background tasks,
and interactive UI for AI-powered watermark removal.
"""

import sys
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from core.config import settings
from api.router import router as api_router, get_service
from ui.app import build_ui


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for model warming, resource initialization,
    and graceful teardown.
    """
    print(f"[*] Bootstrapping {settings.PROJECT_NAME} v{settings.VERSION}...")
    print(f"[*] Default device target: {settings.DEVICE}")

    # Eagerly initialize and warm up model inference engines
    try:
        service = get_service()
        print(f"[*] WatermarkDetector and InpaintingLaMa loaded successfully on: {service.inpainter.device}")
    except Exception as exc:
        print(f"[!] Warning: Model engine lazy initialization fallback: {exc}")

    yield

    print(f"[*] Shutting down {settings.PROJECT_NAME}. Cleanly releasing GPU/CPU resources...")


def create_app() -> FastAPI:
    """Factory to create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Enterprise-grade AI Watermark Removal Platform (FastAPI + LaMa + SAM 2)",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS for enterprise frontend & microservice integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API v1 routes (/upload, /process, /result/{job_id}, /status/{job_id}, /health)
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Mount Gradio interactive interface at /ui
    ui_app = build_ui()
    app = gr.mount_gradio_app(app, ui_app, path="/ui")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info",
    )
