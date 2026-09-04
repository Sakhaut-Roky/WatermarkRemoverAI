"""
Video Watermark Removal Pipeline for WatermarkRemoverAI
======================================================
Frame-by-frame neural video watermark removal with strict memory controls:
1. Enforces strict maximum duration ceiling (10.0 seconds) to prevent VRAM/RAM exhaustion.
2. Streams frames to disk-backed temporary caching to keep memory footprint minimal.
3. Sequential per-frame detection via WatermarkDetector (EasyOCR / SAM 2 / OpenCV)
   and deep inpainting via InpaintingLaMa (Fast Fourier Convolutions / Navier-Stokes).
4. Video reassembly into MP4 container via MoviePy.
5. Strict preservation and synchronization of the original audio track.
6. Real-time progress tracking with callback notifications and metrics.
"""

import os
import time
import inspect
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any, List, Tuple

import numpy as np
import cv2
import torch

try:
    from moviepy import VideoFileClip, ImageSequenceClip
    HAS_MOVIEPY = True
except ImportError:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        HAS_MOVIEPY = True
    except ImportError:
        VideoFileClip = None
        ImageSequenceClip = None
        HAS_MOVIEPY = False

from core.mask_generator import WatermarkDetector
from core.inpaint_engine import InpaintingLaMa

# Structured Logger
logger = logging.getLogger("WatermarkRemoverAI.VideoProcessor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Strict hard ceiling on allowable video duration in seconds to prevent memory overflow
STRICT_MAX_DURATION_SECONDS: float = 10.0


def _subclip_compat(clip: Any, start_sec: float, end_sec: float) -> Any:
    """
    Subclips a video clip with full cross-compatibility across
    MoviePy 1.x (clip.subclip) and MoviePy 2.x (clip.subclipped).
    """
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start_sec, end_sec)
    elif hasattr(clip, "subclip"):
        return clip.subclip(start_sec, end_sec)
    return clip


def _set_audio_compat(clip: Any, audio_clip: Any) -> Any:
    """
    Attaches audio to a video clip with cross-compatibility across
    MoviePy 1.x (clip.set_audio) and MoviePy 2.x (clip.with_audio).
    """
    if audio_clip is None:
        return clip
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    elif hasattr(clip, "set_audio"):
        return clip.set_audio(audio_clip)
    return clip


def _apply_removal_mode(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    mode: str,
    inpainter: InpaintingLaMa,
    composite: bool = True
) -> np.ndarray:
    """
    Executes specified watermark removal algorithm on an RGB frame.

    Supported Modes:
    - "Smooth Edge Interpolation": Bilateral/Navier-Stokes edge-preserving inpainting.
    - "Gaussian Blur Blend": Heavy Gaussian blur with smooth edge alpha-feathering.
    - "Pixelate": Mosaic downsampling and nearest-neighbor reconstruction.
    - "Inpaint (Content-Aware Fill)": Neural LaMa Fast Fourier Convolution inpainting.
    """
    if mode == "Smooth Edge Interpolation":
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        res = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        if composite:
            m = (mask > 0)[:, :, None]
            res = np.where(m, res, image_rgb)
        return res

    elif mode == "Gaussian Blur Blend":
        ksize = (35, 35)
        blurred = cv2.GaussianBlur(image_rgb, ksize, 15)
        feathered_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0)[:, :, None]
        blended = (blurred * feathered_mask + image_rgb * (1.0 - feathered_mask)).clip(0, 255).astype(np.uint8)
        return blended

    elif mode == "Pixelate":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = image_rgb.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            roi = res[y:y+h, x:x+w]
            sw, sh = max(1, w // 10), max(1, h // 10)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            res[y:y+h, x:x+w] = pixelated
        return res

    else:
        # Default: "Inpaint (Content-Aware Fill)" via LaMa
        return inpainter.inpaint(image=image_rgb, mask=mask, composite=composite, is_bgr=False)


class VideoWatermarkRemover:
    """
    Enterprise-grade sequential video watermark removal processor.

    Orchestrates frame extraction, neural watermark detection, LaMa inpainting,
    and MoviePy video reassembly with strict audio track preservation.
    """

    def __init__(
        self,
        detector: Optional[WatermarkDetector] = None,
        inpainter: Optional[InpaintingLaMa] = None,
        device: Optional[str] = None,
        max_duration_seconds: float = STRICT_MAX_DURATION_SECONDS,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the VideoWatermarkRemover pipeline.

        Args:
            detector: WatermarkDetector instance (instantiated if None).
            inpainter: InpaintingLaMa instance (instantiated if None).
            device: Compute device target ('cuda', 'mps', 'cpu').
            max_duration_seconds: Upper limit on video length (capped at strict 10.0s).
            temp_dir: Custom filesystem location for intermediate frame storage.
        """
        if not HAS_MOVIEPY:
            raise ImportError(
                "moviepy is required for video watermark processing. "
                "Install via 'pip install moviepy'."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enforce strict 10-second maximum ceiling
        if max_duration_seconds > STRICT_MAX_DURATION_SECONDS:
            logger.warning(
                "Requested max duration (%.2fs) exceeds the strict %.1fs safety limit. "
                "Capping to %.1f seconds.",
                max_duration_seconds,
                STRICT_MAX_DURATION_SECONDS,
                STRICT_MAX_DURATION_SECONDS,
            )
            self.max_duration_seconds = STRICT_MAX_DURATION_SECONDS
        else:
            self.max_duration_seconds = max(0.5, float(max_duration_seconds))

        self.temp_dir = Path(temp_dir) if temp_dir else None
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.detector = detector or WatermarkDetector(device=self.device, auto_fallback=True)
        self.inpainter = inpainter or InpaintingLaMa(device=self.device)

        logger.info(
            "VideoWatermarkRemover initialized (Device: %s, Max Duration Ceiling: %.1fs)",
            self.device,
            self.max_duration_seconds,
        )

    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extracts structural metadata from a video file.

        Args:
            video_path: Path to target video file.

        Returns:
            Dictionary containing duration, fps, resolution, frame counts, and audio presence.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found at: {path.resolve()}")

        clip = VideoFileClip(str(path))
        try:
            duration = float(clip.duration) if clip.duration is not None else 0.0
            fps = float(clip.fps) if clip.fps is not None else 30.0
            width, height = int(clip.size[0]), int(clip.size[1])
            has_audio = clip.audio is not None
            total_frames = int(round(duration * fps)) if duration > 0 else 0
            clamped_duration = min(duration, self.max_duration_seconds)
            clamped_frames = int(round(clamped_duration * fps))

            return {
                "file_name": path.name,
                "file_path": str(path.resolve()),
                "file_size_bytes": path.stat().st_size,
                "raw_duration_seconds": round(duration, 3),
                "clamped_duration_seconds": round(clamped_duration, 3),
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "resolution": (width, height),
                "has_audio": has_audio,
                "raw_total_frames": total_frames,
                "processable_frames": clamped_frames,
                "duration_exceeded_10s": duration > self.max_duration_seconds,
            }
        finally:
            clip.close()

    def extract_preview_frame(
        self,
        video_path: Union[str, Path],
        timestamp_seconds: float = 0.0
    ) -> np.ndarray:
        """
        Extracts a single representative RGB frame from the video for preview or canvas mask drafting.

        Args:
            video_path: Path to video file.
            timestamp_seconds: Timestamp from which to grab the frame.

        Returns:
            np.ndarray: Frame in RGB format (H, W, 3).
        """
        path = Path(video_path)
        clip = VideoFileClip(str(path))
        try:
            target_t = min(max(0.0, timestamp_seconds), max(0.0, (clip.duration or 1.0) - 0.05))
            frame_rgb = clip.get_frame(target_t)
            return frame_rgb.copy()
        finally:
            clip.close()

    @staticmethod
    def _notify_progress(
        callback: Optional[Callable[..., None]],
        current_frame: int,
        total_frames: int,
        stage: str,
        elapsed_seconds: float,
        message: str = ""
    ) -> None:
        """Dispatches structured progress notification to user callback."""
        if callback is None:
            return

        percent = round((current_frame / total_frames) * 100.0, 1) if total_frames > 0 else 0.0
        fps_speed = round(current_frame / elapsed_seconds, 2) if elapsed_seconds > 0.05 else 0.0
        eta_seconds = round((total_frames - current_frame) / fps_speed, 1) if (fps_speed > 0 and current_frame < total_frames) else 0.0

        payload = {
            "current_frame": current_frame,
            "total_frames": total_frames,
            "percent": percent,
            "stage": stage,
            "fps_speed": fps_speed,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "eta_seconds": eta_seconds,
            "message": message,
        }

        try:
            sig = inspect.signature(callback)
            params_count = len(sig.parameters)
            if params_count == 1:
                callback(payload)
            elif params_count >= 4:
                callback(current_frame, total_frames, percent, stage)
            else:
                callback(percent)
        except Exception as exc:
            logger.debug("Progress callback error: %s", exc)

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        mask: Optional[Union[np.ndarray, str, Path]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        removal_mode: str = "Inpaint (Content-Aware Fill)",
        static_mask: bool = False,
        composite: bool = True,
        progress_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a video file to remove watermarks frame-by-frame with audio preservation.

        Pipeline Workflow:
        1. Opens video and inspects properties (duration, fps, resolution, audio).
        2. Clamps duration to the strict 10.0s maximum limit.
        3. Extracts frames sequentially and pipes them into temporary disk storage.
        4. For each frame:
           a. Applies bounding box mask if supplied, else detects watermark via EasyOCR / SAM 2 / OpenCV.
           b. Inpaints watermark region via selected removal mode ("Smooth Edge Interpolation",
              "Gaussian Blur Blend", "Pixelate", or "Inpaint (Content-Aware Fill)").
           c. Writes clean RGB frame to temporary frame cache.
           d. Reports progress with elapsed time, FPS speed, and ETA.
        5. Reassembles clean frames into MP4 via MoviePy ImageSequenceClip.
        6. Preserves and synchronizes the original audio track.
        7. Releases resources and cleans up temporary files.

        Args:
            video_path: Source video file path.
            output_path: Destination MP4 file path (auto-generated if None).
            mask: Optional custom binary mask (numpy array or image file) applied to all frames.
            bbox: Optional bounding box tuple (x, y, width, height) in image pixel coordinates.
            removal_mode: Removal algorithm ("Smooth Edge Interpolation", "Gaussian Blur Blend",
                          "Pixelate", "Inpaint (Content-Aware Fill)").
            static_mask: If True, computes watermark mask from frame 1 and reuses across all frames.
            composite: If True, preserves unmasked original pixels byte-for-byte.
            progress_callback: Callback function receiving progress updates.

        Returns:
            Dictionary containing detailed processing metrics and output artifact path.
        """
        source_path = Path(video_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Input video file not found: {source_path.resolve()}")

        # Setup output path
        if output_path is None:
            out_file = source_path.parent / f"{source_path.stem}_cleaned.mp4"
        else:
            out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        start_wall_time = time.time()
        logger.info("Opening input video clip: %s", source_path.name)
        clip = VideoFileClip(str(source_path))

        try:
            raw_duration = float(clip.duration) if clip.duration is not None else 0.0
            if raw_duration <= 0.0:
                raise ValueError(f"Invalid video duration ({raw_duration}s) in {source_path.name}")

            effective_fps = float(clip.fps) if clip.fps is not None else 30.0
            orig_width, orig_height = int(clip.size[0]), int(clip.size[1])
            has_audio = (clip.audio is not None)

            # Enforce strict 10.0s ceiling
            effective_duration = min(raw_duration, self.max_duration_seconds)
            if raw_duration > self.max_duration_seconds:
                logger.warning(
                    "Video duration (%.2fs) exceeds the strict %.1fs maximum limit. "
                    "Clamping execution strictly to the first %.1fs to protect memory.",
                    raw_duration,
                    self.max_duration_seconds,
                    self.max_duration_seconds,
                )

            # Subclip matching effective duration
            subclip = _subclip_compat(clip, 0.0, effective_duration)
            subclip_audio = subclip.audio if has_audio else None

            expected_frames = max(1, int(round(effective_duration * effective_fps)))
            logger.info(
                "Processing video range: 0.0s -> %.2fs | Total Expected Frames: %d | FPS: %.2f | Audio: %s | Mode: %s",
                effective_duration,
                expected_frames,
                effective_fps,
                has_audio,
                removal_mode,
            )

            # Pre-parse static mask or bbox if provided by user
            predefined_mask: Optional[np.ndarray] = None
            if bbox is not None:
                bx, by, bw, bh = bbox
                predefined_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                x1 = max(0, min(orig_width - 1, int(bx)))
                y1 = max(0, min(orig_height - 1, int(by)))
                x2 = max(0, min(orig_width, int(bx + bw)))
                y2 = max(0, min(orig_height, int(by + bh)))
                if x2 > x1 and y2 > y1:
                    predefined_mask[y1:y2, x1:x2] = 255
                    logger.info("Custom bounding box mask set: [X:%d->%d, Y:%d->%d, W:%d, H:%d]", x1, x2, y1, y2, x2-x1, y2-y1)
            elif mask is not None:
                if isinstance(mask, (str, Path)):
                    mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        predefined_mask = (mask_img > 127).astype(np.uint8) * 255
                elif isinstance(mask, np.ndarray):
                    m_arr = mask.squeeze()
                    if m_arr.ndim == 2:
                        predefined_mask = (m_arr > 127).astype(np.uint8) * 255

                if predefined_mask is not None and predefined_mask.shape != (orig_height, orig_width):
                    predefined_mask = cv2.resize(
                        predefined_mask,
                        (orig_width, orig_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                logger.info("Custom static watermark mask mounted for all video frames.")

            # Create dedicated temporary workspace for frames
            with tempfile.TemporaryDirectory(dir=self.temp_dir, prefix="wm_video_frames_") as frame_temp_dir:
                frame_temp_path = Path(frame_temp_dir)
                cleaned_frame_paths: List[str] = []

                cached_detected_mask: Optional[np.ndarray] = None
                frames_with_watermark = 0

                logger.info("Extracting and inpainting frames sequentially...")
                self._notify_progress(
                    progress_callback,
                    0,
                    expected_frames,
                    "extracting_and_inpainting",
                    0.0,
                    "Starting sequential frame processing..."
                )

                frame_idx = 0
                for raw_frame in subclip.iter_frames(fps=effective_fps, dtype="uint8"):
                    if frame_idx >= expected_frames:
                        break

                    frame_rgb = raw_frame
                    frame_h, frame_w = frame_rgb.shape[:2]

                    # Watermark Mask Resolution
                    if predefined_mask is not None:
                        frame_mask = predefined_mask
                    elif static_mask and cached_detected_mask is not None:
                        frame_mask = cached_detected_mask
                    else:
                        # Autonomous Watermark Detection (EasyOCR + SAM 2 / OpenCV fallback)
                        frame_mask = self.detector.generate_mask(frame_rgb)
                        if static_mask and cached_detected_mask is None:
                            cached_detected_mask = frame_mask
                            logger.info("Cached static mask detected from keyframe 0.")

                    has_watermark = np.count_nonzero(frame_mask == 255) > 0
                    if has_watermark:
                        frames_with_watermark += 1
                        # Apply selected removal algorithm
                        cleaned_rgb = _apply_removal_mode(
                            image_rgb=frame_rgb,
                            mask=frame_mask,
                            mode=removal_mode,
                            inpainter=self.inpainter,
                            composite=composite
                        )
                    else:
                        cleaned_rgb = frame_rgb

                    # Save cleaned frame to disk cache (MoviePy streams from disk)
                    frame_filename = frame_temp_path / f"frame_{frame_idx:06d}.png"
                    cleaned_bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(frame_filename), cleaned_bgr)
                    cleaned_frame_paths.append(str(frame_filename))

                    frame_idx += 1
                    elapsed = time.time() - start_wall_time

                    # Update progress every frame
                    self._notify_progress(
                        progress_callback,
                        frame_idx,
                        expected_frames,
                        "extracting_and_inpainting",
                        elapsed,
                        f"Processed frame {frame_idx}/{expected_frames}"
                    )

                    # Periodically flush CUDA cache if running on GPU
                    if self.device.startswith("cuda") and frame_idx % 25 == 0:
                        torch.cuda.empty_cache()

                total_frames_extracted = len(cleaned_frame_paths)
                if total_frames_extracted == 0:
                    raise RuntimeError("No frames could be extracted from the specified video clip.")

                logger.info(
                    "Successfully processed %d frames (%d watermarked). Initiating MoviePy video reassembly...",
                    total_frames_extracted,
                    frames_with_watermark,
                )

                # Step 5 & 6: MoviePy Reassembly & Audio Track Preservation
                self._notify_progress(
                    progress_callback,
                    total_frames_extracted,
                    expected_frames,
                    "encoding",
                    time.time() - start_wall_time,
                    "Reassembling MP4 video and synchronizing audio track..."
                )

                cleaned_clip = ImageSequenceClip(cleaned_frame_paths, fps=effective_fps)

                # Preserve and synchronize original audio track
                if subclip_audio is not None:
                    logger.info("Preserving original audio track on reassembled video.")
                    cleaned_clip = _set_audio_compat(cleaned_clip, subclip_audio)

                temp_audio_file = str(frame_temp_path / "temp-audio.m4a") if subclip_audio is not None else None

                # Write final MP4 video file
                logger.info("Writing output video to: %s", out_file.resolve())
                cleaned_clip.write_videofile(
                    str(out_file),
                    fps=effective_fps,
                    codec="libx264",
                    audio_codec="aac" if subclip_audio is not None else None,
                    temp_audiofile=temp_audio_file,
                    remove_temp=True,
                    logger=None,  # Suppress raw MoviePy stdout to keep output clean
                )

                cleaned_clip.close()
                subclip.close()

            total_elapsed_seconds = round(time.time() - start_wall_time, 3)
            avg_fps = round(total_frames_extracted / total_elapsed_seconds, 2) if total_elapsed_seconds > 0 else 0.0

            self._notify_progress(
                progress_callback,
                total_frames_extracted,
                expected_frames,
                "completed",
                total_elapsed_seconds,
                f"Video watermark removal completed in {total_elapsed_seconds}s"
            )

            result_summary = {
                "status": "completed",
                "input_video": str(source_path.resolve()),
                "output_video": str(out_file.resolve()),
                "original_duration_seconds": round(raw_duration, 2),
                "processed_duration_seconds": round(effective_duration, 2),
                "duration_clamped_to_10s": raw_duration > self.max_duration_seconds,
                "fps": round(effective_fps, 2),
                "resolution": (orig_width, orig_height),
                "total_frames_processed": total_frames_extracted,
                "watermarked_frames_count": frames_with_watermark,
                "audio_preserved": has_audio,
                "elapsed_time_seconds": total_elapsed_seconds,
                "average_processing_fps": avg_fps,
                "output_file_size_bytes": out_file.stat().st_size if out_file.is_file() else 0,
            }

            logger.info(
                "Video processing finished: %d frames rendered in %.2fs (%.2f FPS average). Output: %s",
                total_frames_extracted,
                total_elapsed_seconds,
                avg_fps,
                out_file.name,
            )
            return result_summary

        finally:
            clip.close()
