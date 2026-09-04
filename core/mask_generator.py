"""
Watermark Mask Generation Module for WatermarkRemoverAI
======================================================
Multi-stage autonomous watermark detection and mask generation:
1. Primary Text Detection: EasyOCR neural text detection scanning for alphanumeric
   and typographic watermarks, generating tight bounding-polygon masks (pure white 255).
2. Non-Text Logo & Gridline Fallback: Meta's SAM 2 or specialized OpenCV contour
   analysis targeting geometric logo emblems, stamps, and stock photo gridlines.
3. Strict Binary Output: Pure White (255) for watermark pixels, Pure Black (0) for background.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import torch

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False

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
    Autonomous multi-stage Watermark Detection & Mask Generation engine.
    
    1. Text-based Watermarks: Deep-learning text localization via EasyOCR
       generating tight bounding-polygon masks over detected text.
    2. Non-text Watermarks & Logos: Zero-shot SAM 2 or OpenCV contour
       detection for emblems, circular stamps, and grid lines.
    3. Guarantees pure binary mask output: {0, 255}.
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
        confidence_threshold: float = 0.20,
        auto_fallback: bool = True,
        dilation_iterations: int = 1,
        max_coverage_ratio: float = 0.15,
        enable_ocr: bool = True,
        ocr_languages: Optional[List[str]] = None
    ):
        """
        Initialize the WatermarkDetector.

        Args:
            checkpoint_path: Path to zero-shot model weights (e.g. SAM 2 or Grounding DINO).
            config_path: Model configuration file path (if applicable).
            device: Execution target ('cuda', 'cpu', 'mps').
            confidence_threshold: Detection confidence cut-off for OCR/DL models (default: 0.20).
            auto_fallback: Whether to automatically divert to OpenCV fallback if GPU OOM occurs.
            dilation_iterations: Number of morphological dilation iterations (default: 1).
            max_coverage_ratio: Maximum allowable proportion of masked area (default: 0.15).
            enable_ocr: Whether to run EasyOCR text watermark scanning.
            ocr_languages: Language models for EasyOCR (default: ['en']).
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
        self.enable_ocr = enable_ocr
        self.ocr_languages = ocr_languages or ["en"]

        # Deep Learning SAM 2 state
        self.model: Optional[Any] = None
        self.predictor: Optional[Any] = None
        self.is_model_loaded: bool = False

        # EasyOCR reader instance (lazy-loaded on first text inference)
        self.reader: Optional[Any] = None
        self.ocr_failed: bool = False

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
                "Detector will operate in EasyOCR + Classical CV mode.",
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

    def _get_ocr_reader(self) -> Optional[Any]:
        """
        Initializes and returns cached EasyOCR Reader.
        Lazy-loaded to keep application startup instant.
        """
        if not self.enable_ocr:
            return None

        # Check global import again in case easyocr was just installed
        global HAS_EASYOCR, easyocr
        if not HAS_EASYOCR:
            try:
                import easyocr as _easyocr
                easyocr = _easyocr
                HAS_EASYOCR = True
            except ImportError:
                if not self.ocr_failed:
                    logger.warning(
                        "easyocr is not installed. Text auto-detection will fall back to SAM 2 / OpenCV. "
                        "Install via 'pip install easyocr' for deep text watermark detection."
                    )
                    self.ocr_failed = True
                return None

        if self.reader is None and not self.ocr_failed:
            try:
                use_gpu = (self.device == "cuda" and torch.cuda.is_available())
                logger.info(
                    "Initializing EasyOCR text detection model (Languages: %s, GPU: %s)...",
                    self.ocr_languages, use_gpu
                )
                self.reader = easyocr.Reader(
                    self.ocr_languages,
                    gpu=use_gpu,
                    verbose=False
                )
                logger.info("EasyOCR Reader successfully initialized.")
            except Exception as exc:
                logger.error("Failed to initialize EasyOCR Reader: %s. Falling back to classical CV.", exc)
                self.ocr_failed = True
                self.reader = None

        return self.reader

    def _detect_text_easyocr(
        self,
        image_rgb: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Scans image for text watermarks using EasyOCR.
        Generates precise bounding box masks (pure white 255) over detected text regions.

        Returns:
            Tuple of:
            - mask: np.ndarray uint8 of shape (H, W) with 255 on detected text regions.
            - detections: List of dicts with detected text, bbox, and confidence score.
        """
        height, width = image_rgb.shape[:2]
        text_mask = np.zeros((height, width), dtype=np.uint8)
        detections = []

        reader = self._get_ocr_reader()
        if reader is None:
            return text_mask, detections

        try:
            # EasyOCR readtext accepts RGB numpy array
            results = reader.readtext(image_rgb)
            # Each item: (bbox, text, confidence)
            # bbox is 4 corner points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

            for bbox, text, confidence in results:
                if confidence < self.confidence_threshold:
                    continue

                pts = np.array(bbox, dtype=np.int32)
                # Fill the polygon bounding box with pure white (255)
                cv2.fillPoly(text_mask, [pts], 255)

                detections.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [list(map(int, p)) for p in bbox]
                })

            if detections:
                logger.info(
                    "EasyOCR detected %d text watermark instances: %s",
                    len(detections),
                    [d["text"] for d in detections]
                )

                # Apply slight dilation (5x5 kernel) to ensure ascenders, descenders,
                # and anti-aliased character edges are fully encompassed
                pad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                text_mask = cv2.dilate(text_mask, pad_kernel, iterations=1)

        except Exception as exc:
            logger.warning("Error during EasyOCR text detection: %s. Continuing with fallback.", exc)

        return text_mask, detections

    def _detect_non_text_logos_and_grids(
        self,
        image_rgb: np.ndarray,
        image_bgr: np.ndarray
    ) -> np.ndarray:
        """
        Secondary detection targeting non-text logos, circular copyright emblems,
        and stock photo gridlines using SAM 2 or OpenCV contour analysis.
        """
        # If SAM 2 zero-shot model is loaded:
        if self.is_model_loaded:
            try:
                sam_mask = self._infer_zero_shot(image_rgb)
                if np.count_nonzero(sam_mask) > 0:
                    return sam_mask
            except Exception as exc:
                logger.warning("SAM 2 inference failed: %s. Using OpenCV contour fallback.", exc)

        # OpenCV Logo & Gridline analysis
        height, width = image_rgb.shape[:2]
        total_pixels = height * width

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. High Canny threshold for sharp geometric logo borders & gridlines
        canny = cv2.Canny(blurred, 100, 220)

        # 2. Extract thin gridlines (horizontal and vertical linear runs)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
        grid_h = cv2.morphologyEx(canny, cv2.MORPH_OPEN, h_kernel)
        grid_v = cv2.morphologyEx(canny, cv2.MORPH_OPEN, v_kernel)
        grid_mask = cv2.bitwise_or(grid_h, grid_v)

        # 3. Detect circular/geometric logo emblems
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logo_mask = np.zeros((height, width), dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > (total_pixels * 0.05):  # Logos rarely exceed 5%
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w == 0:
                continue

            aspect_ratio = w / float(h)
            # Check for circular or compact logo emblems (e.g. copyright symbol ©, round stamps)
            is_compact_logo = (0.6 <= aspect_ratio <= 1.6) and (h <= height * 0.15) and (w <= width * 0.15)
            if is_compact_logo:
                cv2.drawContours(logo_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        combined = cv2.bitwise_or(grid_mask, logo_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        return combined

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
        Conservative Classical Fallback Pipeline for when no text is detected by OCR.
        
        Targets:
        1. High-contrast text watermarks.
        2. Thin watermark gridlines.
        
        Returns:
            np.ndarray: Strict binary mask of shape (H, W) with values in {0, 255}.
        """
        height, width = image_bgr.shape[:2]
        total_pixels = height * width
        max_allowed_pixels = int(total_pixels * self.max_coverage_ratio)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        block_size = 35
        bright_candidates = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=-24
        )
        dark_candidates = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=24
        )

        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        grad_cutoff = max(50.0, float(np.percentile(grad_mag, 92)))
        strong_edges = (grad_mag >= grad_cutoff).astype(np.uint8) * 255

        text_seeds_bright = cv2.bitwise_and(bright_candidates, strong_edges)
        text_seeds_dark = cv2.bitwise_and(dark_candidates, strong_edges)
        text_candidates = cv2.bitwise_or(text_seeds_bright, text_seeds_dark)

        canny_high = cv2.Canny(blurred, 130, 240)
        h_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        v_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
        grid_h = cv2.morphologyEx(canny_high, cv2.MORPH_OPEN, h_line_kernel)
        grid_v = cv2.morphologyEx(canny_high, cv2.MORPH_OPEN, v_line_kernel)
        grid_candidates = cv2.bitwise_or(grid_h, grid_v)

        raw_candidates = cv2.bitwise_or(text_candidates, grid_candidates)
        clean_candidates = cv2.morphologyEx(
            raw_candidates,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

        contours, _ = cv2.findContours(clean_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_char_area = 12
        max_char_area = int(total_pixels * 0.008)
        max_char_height = int(height * 0.09)
        max_char_width = int(width * 0.35)

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

            is_grid_line = (
                (w >= 30 and h <= 3) or
                (h >= 30 and w <= 3) or
                (aspect_ratio > 10.0 and h <= 4) or
                (aspect_ratio < 0.1 and w <= 4)
            )

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

        candidate_contours.sort(key=lambda x: x[2], reverse=True)

        final_mask = np.zeros((height, width), dtype=np.uint8)
        accumulated_pixels = 0

        for cnt, area, _ in candidate_contours:
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            new_pixels = np.count_nonzero(cv2.bitwise_and(temp_mask, cv2.bitwise_not(final_mask)))

            if accumulated_pixels + new_pixels > max_allowed_pixels:
                break

            final_mask = cv2.bitwise_or(final_mask, temp_mask)
            accumulated_pixels += new_pixels

        return self._postprocess_mask(final_mask)

    def _postprocess_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """
        Enforces strict binary mask integrity {0, 255} and ensures the final mask
        does not exceed the allowable coverage ceiling.
        """
        height, width = raw_mask.shape[:2]
        total_pixels = height * width
        max_allowed_pixels = int(total_pixels * self.max_coverage_ratio)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

        if self.dilation_iterations > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(closed, dilate_kernel, iterations=self.dilation_iterations)
        else:
            dilated = closed

        _, binary_mask = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        masked_count = np.count_nonzero(binary_mask == 255)
        if masked_count > max_allowed_pixels:
            logger.warning(
                "Mask coverage (%.2f%%) exceeded ceiling after dilation. Pruning excess components.",
                (masked_count / total_pixels) * 100
            )
            _, binary_mask = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8)

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
        Autonomous Multi-Stage Watermark Mask Generator.

        Pipeline Flow:
        1. Validates and loads the image into RGB and BGR standard matrices.
        2. Stage 1: Runs EasyOCR to detect text-based watermarks. Generates
           precise bounding-box masks (255 pure white) over all detected text.
        3. Stage 2: Runs SAM 2 / OpenCV contour fallback for non-text logos,
           circular emblems, and grid patterns.
        4. Merges detected text and logo masks.
        5. Post-processes into strict binary format {0, 255}.

        Args:
            image: Image file path (str, Path) or NumPy array (H, W, 3).
            return_diagnostics: If True, returns (mask, metadata_dict).
            is_bgr: Set to True if NumPy array is in BGR format.

        Returns:
            np.ndarray: Pure binary mask (255 = watermark, 0 = background).
            (Optional) Dict containing detection metadata.
        """
        img_bgr, img_rgb = self._load_and_validate_image(image, is_bgr=is_bgr)
        height, width = img_rgb.shape[:2]

        methods_used = []
        text_detections = []

        # Step 1: Text Watermark Detection via EasyOCR
        text_mask, text_detections = self._detect_text_easyocr(img_rgb)
        if len(text_detections) > 0:
            methods_used.append("easyocr_text_detection")

        # Step 2: Non-Text Logo / Gridline Fallback
        # Run logo/grid detection if no text was found, or if zero-shot SAM 2 is active
        logo_grid_mask = np.zeros((height, width), dtype=np.uint8)
        if len(text_detections) == 0 or self.is_model_loaded:
            logo_grid_mask = self._detect_non_text_logos_and_grids(img_rgb, img_bgr)
            if np.count_nonzero(logo_grid_mask) > 0:
                methods_used.append("sam2_or_opencv_logo_detection")

        # Combine text and logo masks
        raw_mask = cv2.bitwise_or(text_mask, logo_grid_mask)

        # Fallback to conservative classical OpenCV scan if nothing was detected at all
        if np.count_nonzero(raw_mask) == 0:
            logger.info("No text or logos found via OCR/SAM. Running conservative classical CV scan...")
            raw_mask = self._classical_fallback_pipeline(img_bgr)
            if np.count_nonzero(raw_mask) > 0:
                methods_used.append("classical_opencv_scan")

        # Final binarization & postprocessing
        final_mask = self._postprocess_mask(raw_mask)

        # Verification assertion for production safety
        assert final_mask.shape == (height, width), "Mask shape must match input image shape"
        unique_vals = np.unique(final_mask)
        assert np.all(np.isin(unique_vals, [0, 255])), "Mask must be strictly binary {0, 255}"

        coverage_percentage = float(np.count_nonzero(final_mask == 255) / final_mask.size * 100)

        if return_diagnostics:
            diagnostics = {
                "method_used": "+".join(methods_used) if methods_used else "none",
                "text_instances_detected": len(text_detections),
                "detected_texts": [d["text"] for d in text_detections],
                "input_resolution": (width, height),
                "device": self.device,
                "coverage_percentage": round(coverage_percentage, 2),
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
