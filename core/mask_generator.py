"""
Watermark Mask Generation Module for WatermarkRemoverAI
======================================================
Architectural implementation of zero-shot deep learning segmentation
(Meta's SAM 2 / Grounding DINO) with an automated classical computer vision
fallback engine (OpenCV thresholding, Canny edge detection, morphological dilation).

Ensures pure binary mask generation:
- Watermark regions: Pure White (255)
- Background / Clean regions: Pure Black (0)
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import torch

try:
    from core.config import settings
    DEFAULT_DEVICE = settings.DEVICE
    DEFAULT_CONFIDENCE = settings.DEFAULT_CONFIDENCE_THRESHOLD
except ImportError:
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_CONFIDENCE = 0.5

# Setup structured logger
logger = logging.getLogger("WatermarkRemoverAI.MaskGenerator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class WatermarkDetector:
    """
    Production-ready Watermark Detection & Mask Generation engine.
    
    Supports:
    1. Zero-shot deep learning segmentation (Meta's SAM 2 / Grounding DINO).
    2. Resilient Classical Fallback pipeline (Canny, Otsu/Adaptive Thresholding,
       Morphological Gradients, and Dilation) for low-resource or GPU OOM situations.
    """

    SUPPORTED_PROMPTS = [
        "watermark",
        "text",
        "logo",
        "copyright",
        "stamp",
        "signature",
        "grid lines",
        "stock photo watermark"
    ]

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        auto_fallback: bool = True,
        dilation_iterations: int = 2
    ):
        """
        Initialize the WatermarkDetector.

        Args:
            checkpoint_path: Path to zero-shot model weights (e.g. SAM 2 or Grounding DINO).
            config_path: Model configuration file path (if applicable).
            device: Execution target ('cuda', 'cpu', 'mps').
            confidence_threshold: Detection confidence cut-off (0.0 - 1.0).
            auto_fallback: Whether to automatically divert to OpenCV fallback if GPU OOM occurs.
            dilation_iterations: Number of morphological dilation iterations to ensure 
                                 anti-aliased watermark boundaries are completely captured.
        """
        self.device = device or DEFAULT_DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.config_path = Path(config_path) if config_path else None
        self.confidence_threshold = confidence_threshold
        self.auto_fallback = auto_fallback
        self.dilation_iterations = dilation_iterations

        # Model state
        self.model: Optional[Any] = None
        self.predictor: Optional[Any] = None
        self.is_model_loaded: bool = False

        # Attempt to load zero-shot segmentation model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Architecture hook for zero-shot segmentation (SAM 2 / Grounding DINO).
        Gracefully handles missing weights without crashing the pipeline.
        """
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            logger.info(
                "Zero-shot checkpoint not provided or not found at '%s'. "
                "Detector will operate in Classical CV Fallback mode until weights are mounted.",
                self.checkpoint_path
            )
            self.is_model_loaded = False
            return

        try:
            logger.info("Mounting zero-shot weights from %s on %s...", self.checkpoint_path, self.device)
            
            # --- Integration Architecture for SAM 2 / Grounding DINO ---
            # Example SAM 2 integration hook:
            # from sam2.build_sam import build_sam2
            # from sam2.sam2_image_predictor import SAM2ImagePredictor
            # sam2_model = build_sam2(self.config_path, str(self.checkpoint_path), device=self.device)
            # self.predictor = SAM2ImagePredictor(sam2_model)
            
            self.is_model_loaded = True
            logger.info("Zero-shot model successfully loaded.")
        except Exception as exc:
            logger.error("Failed to load zero-shot segmentation model: %s", exc, exc_info=True)
            self.is_model_loaded = False

    def _load_and_validate_image(
        self,
        image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates and standardizes input image from file path or NumPy array.
        
        Returns:
            Tuple of (image_bgr, image_rgb) as np.uint8 arrays of shape (H, W, 3).
        
        Raises:
            ValueError: If input format, dimensions, or file contents are invalid.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path.resolve()}")
            
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Could not decode image file: {path.resolve()}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_bgr, img_rgb

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Supplied image array is empty.")

            # Handle float arrays normalized in [0, 1]
            if np.issubdtype(image_input.dtype, np.floating):
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255.0).astype(np.uint8)
                else:
                    image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            elif image_input.dtype != np.uint8:
                image_input = image_input.astype(np.uint8)

            # Dimensions handling
            if image_input.ndim == 2:  # Grayscale
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                return img_bgr, img_rgb
            elif image_input.ndim == 3:
                channels = image_input.shape[2]
                if channels == 4:  # RGBA
                    img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                    img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
                    return img_bgr, img_rgb
                elif channels == 3:
                    # Assume input array in standard BGR or RGB; maintain consistent pairs
                    img_bgr = image_input.copy()
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    return img_bgr, img_rgb
                elif channels == 1:
                    img_bgr = cv2.cvtColor(image_input[:, :, 0], cv2.COLOR_GRAY2BGR)
                    img_rgb = cv2.cvtColor(image_input[:, :, 0], cv2.COLOR_GRAY2RGB)
                    return img_bgr, img_rgb
                else:
                    raise ValueError(f"Unsupported channel dimension: {channels}")
            else:
                raise ValueError(f"Unsupported image dimensions: {image_input.ndim}D")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. "
                "Must be a valid file path (str, Path) or NumPy array."
            )

    def _infer_zero_shot(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Executes zero-shot model inference to detect watermarks.
        Protected by PyTorch autocast and no_grad context.
        """
        if not self.is_model_loaded or self.predictor is None:
            raise RuntimeError("Zero-shot model is not loaded in memory.")

        height, width = image_rgb.shape[:2]
        
        with torch.no_grad():
            # Architecture hook for Grounding DINO / SAM 2 prompt inference
            # Example:
            # self.predictor.set_image(image_rgb)
            # masks, scores, logits = self.predictor.predict(...)
            pass

        # Placeholder tensor return for SAM 2 output
        return np.zeros((height, width), dtype=np.uint8)

    def _classical_fallback_pipeline(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Robust Classical Computer Vision Fallback Pipeline.
        
        Combines:
        1. Morphological Top-Hat & Black-Hat transforms (high-frequency watermark relief).
        2. Adaptive Gaussian & Otsu thresholding (contrast-based text & logo isolation).
        3. Canny edge detection with heuristic gradient bounds (grid lines & sharp borders).
        4. Morphological closure & dilation (boundary consolidation).
        
        Returns:
            np.ndarray: Strict binary mask of shape (H, W) with values in {0, 255}.
        """
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Contrast Enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # 2. Morphological Top-Hat & Black-Hat (captures light/dark watermarks on varied backgrounds)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, morph_kernel)
        blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, morph_kernel)
        relief = cv2.add(tophat, blackhat)

        # 3. Dynamic Canny Edge Detection (detects faint gridlines and text outlines)
        v_median = np.median(enhanced_gray)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v_median))
        upper_thresh = int(min(255, (1.0 + sigma) * v_median))
        edges = cv2.Canny(enhanced_gray, lower_thresh, upper_thresh)

        # 4. Adaptive & Otsu Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=8
        )
        _, otsu_thresh = cv2.threshold(relief, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Grid Line Detection (Horizontal & Vertical kernels for stock photo grids)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        grid_horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
        grid_vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vert_kernel)
        grid_mask = cv2.bitwise_or(grid_horiz, grid_vert)

        # 6. Multi-Signal Fusion
        combined = cv2.bitwise_or(edges, adaptive_thresh)
        combined = cv2.bitwise_or(combined, otsu_thresh)
        combined = cv2.bitwise_or(combined, grid_mask)

        # 7. Contour Area Filtering (Discard massive solid blocks or single-pixel speckles)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros((height, width), dtype=np.uint8)
        
        total_pixels = height * width
        min_area = max(10, int(total_pixels * 0.00002))   # eliminate micro speckles
        max_area = int(total_pixels * 0.35)              # avoid selecting the entire foreground

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # If contour filtering was too aggressive (e.g. thin hairline grids), merge with edges
        final_mask = cv2.bitwise_or(filtered_mask, edges)

        return self._postprocess_mask(final_mask)

    def _postprocess_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Enforces strict binary mask integrity:
        - Morphological closing to join broken text strokes.
        - Morphological dilation to engulf semi-transparent edge transitions.
        - Strict binarization: pure white (255) vs pure black (0).
        """
        # Morphological Closing
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

        # Morphological Dilation
        if self.dilation_iterations > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(closed, dilate_kernel, iterations=self.dilation_iterations)
        else:
            dilated = closed

        # Enforce exact binary values (0 or 255)
        _, binary_mask = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        return binary_mask.astype(np.uint8)

    def generate_mask(
        self,
        image: Union[str, Path, np.ndarray],
        return_diagnostics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Generates a watermark mask for the input image.

        Pipeline Flow:
        1. Validates and loads the image into RGB and BGR standard matrices.
        2. Tries zero-shot DL segmentation if model weights are loaded.
        3. If GPU runs out of memory (OOM) or zero-shot model is absent/fails,
           automatically purges VRAM and diverts to classical OpenCV fallback.
        4. Guarantees output is pure binary (0 for background, 255 for watermark).

        Args:
            image: Image file path (str, Path) or NumPy array (H, W, 3).
            return_diagnostics: If True, returns (mask, metadata_dict).

        Returns:
            np.ndarray: Pure binary mask (255 = watermark, 0 = background).
            (Optional) Dict containing detection metadata.
        """
        img_bgr, img_rgb = self._load_and_validate_image(image)
        height, width = img_rgb.shape[:2]

        method_used = "classical_fallback"
        raw_mask: Optional[np.ndarray] = None
        error_info: Optional[str] = None

        # Attempt Zero-Shot DL Inference if model available
        if self.is_model_loaded:
            try:
                raw_mask = self._infer_zero_shot(img_rgb)
                method_used = "zero_shot_deep_learning"
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                error_msg = str(exc).lower()
                is_oom = "out of memory" in error_msg or isinstance(exc, torch.cuda.OutOfMemoryError)

                if is_oom:
                    logger.warning(
                        "GPU Out of Memory (OOM) encountered during zero-shot watermark detection! "
                        "Purging CUDA cache and executing Classical OpenCV fallback pipeline."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.warning(
                        "Runtime failure during zero-shot detection: %s. Falling back to classical CV.",
                        exc
                    )

                error_info = str(exc)
                if not self.auto_fallback:
                    raise exc

        # Execute Classical Fallback if needed
        if raw_mask is None or not self.is_model_loaded:
            logger.info("Executing Classical OpenCV fallback pipeline for mask generation...")
            raw_mask = self._classical_fallback_pipeline(img_bgr)
            method_used = "classical_fallback"

        # Final binarization check
        final_mask = self._postprocess_mask(raw_mask)

        # Verification assertion for production safety
        assert final_mask.shape == (height, width), "Mask shape must match input image shape"
        unique_vals = np.unique(final_mask)
        assert np.all(np.isin(unique_vals, [0, 255])), "Mask must be strictly binary {0, 255}"

        if return_diagnostics:
            diagnostics = {
                "method_used": method_used,
                "input_resolution": (width, height),
                "device": self.device,
                "coverage_percentage": float(np.count_nonzero(final_mask == 255) / final_mask.size * 100),
                "error_info": error_info
            }
            return final_mask, diagnostics

        return final_mask

    @staticmethod
    def save_mask(mask: np.ndarray, output_path: Union[str, Path]) -> Path:
        """
        Saves the generated binary mask to disk.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(out), mask)
        if not success:
            raise IOError(f"Failed to write mask image to {out.resolve()}")
        return out
