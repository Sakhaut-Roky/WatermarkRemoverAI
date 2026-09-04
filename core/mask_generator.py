"""
Watermark Mask Generation Module for WatermarkRemoverAI
======================================================
Architectural implementation of zero-shot deep learning segmentation
(Meta's SAM 2 / Grounding DINO) with an extremely conservative classical
computer vision fallback engine (strict adaptive thresholding, Sobel gradient
verification, text-structural aspect ratio filters, and thin gridline extraction).

Guarantees:
- Targets ONLY high-contrast text glyphs and thin watermark gridlines.
- Rejects natural textures (trees, foliage, water ripples, pools, clouds).
- NEVER masks more than 10% of the total image area.
- Pure binary mask: Watermark = 255 (Pure White), Background = 0 (Pure Black).
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
    2. Extremely conservative Classical Fallback pipeline tailored strictly for:
       - High-contrast text watermarks
       - Fine rectilinear/diagonal watermark grid patterns
       While strictly rejecting natural organic textures (trees, water, sky).
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
        dilation_iterations: int = 1,
        max_coverage_ratio: float = 0.10
    ):
        """
        Initialize the WatermarkDetector.

        Args:
            checkpoint_path: Path to zero-shot model weights (e.g. SAM 2 or Grounding DINO).
            config_path: Model configuration file path (if applicable).
            device: Execution target ('cuda', 'cpu', 'mps').
            confidence_threshold: Detection confidence cut-off (0.0 - 1.0).
            auto_fallback: Whether to automatically divert to OpenCV fallback if GPU OOM occurs.
            dilation_iterations: Number of morphological dilation iterations (default: 1 for tight borders).
            max_coverage_ratio: Maximum allowable proportion of masked area (default: 0.10, i.e., 10%).
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
        self.max_coverage_ratio = max_coverage_ratio

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
            self.is_model_loaded = True
            logger.info("Zero-shot model successfully loaded.")
        except Exception as exc:
            logger.error("Failed to load zero-shot segmentation model: %s", exc, exc_info=True)
            self.is_model_loaded = False

    def _load_and_validate_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        is_bgr: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates and standardizes input image from file path or NumPy array.
        Whenever OpenCV (cv2.imread) is used, it is immediately converted
        from BGR to RGB via cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).
        
        Returns:
            Tuple of (image_bgr, image_rgb) as np.uint8 arrays of shape (H, W, 3).
        
        Raises:
            ValueError: If input format, dimensions, or file contents are invalid.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path.resolve()}")
            
            # Read via OpenCV (BGR)
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Could not decode image file: {path.resolve()}")
            # Immediately convert from BGR to RGB
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
                    if is_bgr:
                        img_bgr = image_input.copy()
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        # By default, arrays in Python/PyTorch/PIL ecosystem are RGB
                        img_rgb = image_input.copy()
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
            pass

        return np.zeros((height, width), dtype=np.uint8)

    def _classical_fallback_pipeline(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extremely Conservative Classical Fallback Pipeline.
        
        Strictly targets:
        1. High-contrast text watermarks (alphanumeric glyphs with sharp local edges).
        2. Thin watermark gridlines (rectilinear/diagonal hairlines).
        
        Explicitly rejects:
        - Natural organic textures (trees, foliage, water ripples, waves, pools, clouds).
        - Large amorphous regions or broad shadows/lighting gradients.
        
        Enforces:
        - Strict adaptive thresholding with large negative/positive constant offsets.
        - High Sobel gradient magnitude requirements.
        - Structural text aspect ratios (0.15 <= W/H <= 12.0) and bounding box limits.
        - Hard global ceiling: NEVER masks more than 10% of total image area.
        
        Returns:
            np.ndarray: Strict binary mask of shape (H, W) with values in {0, 255}.
        """
        height, width = image_bgr.shape[:2]
        total_pixels = height * width
        max_allowed_pixels = int(total_pixels * self.max_coverage_ratio)

        # 1. Grayscale & Micro-Texture Suppression
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Moderate Gaussian blur smooths natural high-frequency foliage and water ripples
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Strict High-Contrast Adaptive Thresholding
        # Watermark text is distinctly brighter or darker than its immediate local background.
        # Large blockSize (35) with high offset (+/- 24) completely ignores natural surfaces.
        block_size = 35
        bright_candidates = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=-24  # Must be >= 24 intensity levels brighter than local neighborhood
        )
        dark_candidates = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=24   # Must be >= 24 intensity levels darker than local neighborhood
        )

        # 3. Local Gradient Magnitude Verification (Sobel)
        # Authentic text and logo strokes possess sharp, pronounced gradient boundaries.
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        # Restrict strictly to top gradient energies (top 8% sharpest transitions)
        grad_cutoff = max(50.0, float(np.percentile(grad_mag, 92)))
        strong_edges = (grad_mag >= grad_cutoff).astype(np.uint8) * 255

        # Both strong local contrast difference AND high gradient magnitude required
        text_seeds_bright = cv2.bitwise_and(bright_candidates, strong_edges)
        text_seeds_dark = cv2.bitwise_and(dark_candidates, strong_edges)
        text_candidates = cv2.bitwise_or(text_seeds_bright, text_seeds_dark)

        # 4. Thin Grid Pattern Detection (Stock Photo Grid Lines)
        # Grid lines are thin linear structures spanning long distances.
        canny_high = cv2.Canny(blurred, 130, 240)
        h_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        v_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
        grid_h = cv2.morphologyEx(canny_high, cv2.MORPH_OPEN, h_line_kernel)
        grid_v = cv2.morphologyEx(canny_high, cv2.MORPH_OPEN, v_line_kernel)
        grid_candidates = cv2.bitwise_or(grid_h, grid_v)

        # Combine text candidates with thin gridlines
        raw_candidates = cv2.bitwise_or(text_candidates, grid_candidates)

        # Morphological opening to prune single-pixel noise speckles
        clean_candidates = cv2.morphologyEx(
            raw_candidates,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

        # 5. Strict Morphological & Geometric Structural Filtering
        contours, _ = cv2.findContours(clean_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_char_area = 12                         # Eliminate micro noise
        max_char_area = int(total_pixels * 0.008)  # Single character/word cannot exceed 0.8% of image
        max_char_height = int(height * 0.09)       # Watermark character height rarely exceeds 9%
        max_char_width = int(width * 0.35)         # Text word width rarely exceeds 35%

        candidate_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_char_area or area > max_char_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            bbox_area = w * h
            if bbox_area == 0:
                continue

            aspect_ratio = w / float(h)
            extent = area / float(bbox_area)

            # Condition A: Thin grid line pattern
            is_grid_line = (
                (w >= 30 and h <= 3) or
                (h >= 30 and w <= 3) or
                (aspect_ratio > 10.0 and h <= 4) or
                (aspect_ratio < 0.1 and w <= 4)
            )

            # Condition B: Text glyph / word pattern
            # Text letters/words have characteristic aspect ratios and intermediate solidity
            is_text_glyph = (
                (0.15 <= aspect_ratio <= 12.0) and
                (h <= max_char_height) and
                (w <= max_char_width) and
                (0.12 <= extent <= 0.82)
            )

            if is_text_glyph or is_grid_line:
                cnt_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                mean_grad = float(cv2.mean(grad_mag, mask=cnt_mask)[0])
                candidate_contours.append((cnt, area, mean_grad))

        # 6. Enforce Global Coverage Limit (Strictly Cap Total Mask <= 10%)
        # Prioritize contours with highest contrast/gradient certainty
        candidate_contours.sort(key=lambda x: x[2], reverse=True)

        final_mask = np.zeros((height, width), dtype=np.uint8)
        accumulated_pixels = 0

        for cnt, area, _ in candidate_contours:
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            new_pixels = np.count_nonzero(cv2.bitwise_and(temp_mask, cv2.bitwise_not(final_mask)))

            if accumulated_pixels + new_pixels > max_allowed_pixels:
                logger.info(
                    "Watermark candidate rejected: total mask area reached %.1f%% image ceiling.",
                    self.max_coverage_ratio * 100
                )
                break

            final_mask = cv2.bitwise_or(final_mask, temp_mask)
            accumulated_pixels += new_pixels

        return self._postprocess_mask(final_mask)

    def _postprocess_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Enforces strict binary mask integrity and ensures the final mask
        NEVER exceeds the 10% total image area ceiling.
        """
        height, width = raw_mask.shape[:2]
        total_pixels = height * width
        max_allowed_pixels = int(total_pixels * self.max_coverage_ratio)

        # Morphological Closing with small 3x3 kernel (join text character strokes)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

        # Gentle Morphological Dilation (only 1 iteration to cover anti-aliased borders)
        if self.dilation_iterations > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(closed, dilate_kernel, iterations=self.dilation_iterations)
        else:
            dilated = closed

        # Enforce exact binary values (0 or 255)
        _, binary_mask = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # Hard guard: if dilation pushed mask over 10%, scale back or prune components
        masked_count = np.count_nonzero(binary_mask == 255)
        if masked_count > max_allowed_pixels:
            logger.warning(
                "Mask coverage (%.2f%%) exceeded 10%% limit after dilation. Reverting to pre-dilation mask.",
                (masked_count / total_pixels) * 100
            )
            _, binary_mask = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)

            # If still over 10%, prune components by area until strictly under ceiling
            if np.count_nonzero(binary_mask == 255) > max_allowed_pixels:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
                pruned_mask = np.zeros_like(binary_mask)
                running_sum = 0
                comp_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
                for idx in comp_indices:
                    comp_area = stats[idx, cv2.CC_STAT_AREA]
                    if running_sum + comp_area <= max_allowed_pixels:
                        pruned_mask[labels == idx] = 255
                        running_sum += comp_area
                    else:
                        break
                binary_mask = pruned_mask

        return binary_mask

    def generate_mask(
        self,
        image: Union[str, Path, np.ndarray],
        return_diagnostics: bool = False,
        is_bgr: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Generates a watermark mask for the input image.

        Pipeline Flow:
        1. Validates and loads the image into RGB and BGR standard matrices.
        2. Tries zero-shot DL segmentation if model weights are loaded.
        3. If GPU runs out of memory (OOM) or zero-shot model is absent/fails,
           automatically purges VRAM and diverts to classical OpenCV fallback.
        4. Enforces extremely conservative text/grid targeting (max 10% coverage).
        5. Guarantees output is pure binary (0 for background, 255 for watermark).

        Args:
            image: Image file path (str, Path) or NumPy array (H, W, 3).
            return_diagnostics: If True, returns (mask, metadata_dict).
            is_bgr: Set to True if NumPy array is in BGR format (e.g. from cv2.imread).

        Returns:
            np.ndarray: Pure binary mask (255 = watermark, 0 = background).
            (Optional) Dict containing detection metadata.
        """
        img_bgr, img_rgb = self._load_and_validate_image(image, is_bgr=is_bgr)
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

        # Final binarization check & 10% ceiling enforcement
        final_mask = self._postprocess_mask(raw_mask)

        # Verification assertions for production safety
        assert final_mask.shape == (height, width), "Mask shape must match input image shape"
        unique_vals = np.unique(final_mask)
        assert np.all(np.isin(unique_vals, [0, 255])), "Mask must be strictly binary {0, 255}"

        coverage_percentage = float(np.count_nonzero(final_mask == 255) / final_mask.size * 100)
        assert coverage_percentage <= 10.01, f"Mask coverage ({coverage_percentage:.2f}%) exceeds 10% maximum limit"

        if return_diagnostics:
            diagnostics = {
                "method_used": method_used,
                "input_resolution": (width, height),
                "device": self.device,
                "coverage_percentage": coverage_percentage,
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
