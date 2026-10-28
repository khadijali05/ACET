"""Smoke color analysis using the existing Florence-2 caption model and pixel heuristic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
COLOR_PATTERN = re.compile(r"\b(black|white|grey|gray|blue)\b\s+(smoke|exhaust|fumes)\b", re.IGNORECASE)
COLOR_LABELS = {
    "black": "black_smoke",
    "white": "white_smoke",
    "grey": "grey_smoke",
    "gray": "grey_smoke",
    "blue": "blue_smoke",
}


class ClassifierLoadError(RuntimeError):
    """Raised when the existing Florence-2 model cannot be loaded."""


@dataclass(frozen=True)
class ColorResult:
    label: str | None
    status: Literal["classified", "uncertain", "unavailable"]
    heuristic_score: float
    confidence_kind: str
    florence_caption: str | None
    pixel_shares: dict[str, float]
    explanation: str


class SmokeColorClassifier:
    """Use Florence-2 captions as supporting evidence and report pixel heuristic separately."""

    def __init__(
        self,
        project_root: str | Path,
        device: str | torch.device | None = None,
        min_pixel_share: float = 0.30,
    ) -> None:
        if not 0.0 <= min_pixel_share <= 1.0:
            raise ValueError("min_pixel_share must be between 0 and 1.")
        self.project_root = Path(project_root).resolve()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.min_pixel_share = min_pixel_share
        self.model: AutoModelForCausalLM | None = None
        self.processor: AutoProcessor | None = None
        self.cache_dir = self.project_root / "models" / "florence2_cache"

    def load(self) -> None:
        """Load the existing Florence-2 checkpoint from the Hugging Face cache or download it."""
        if self.model is not None and self.processor is not None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=str(self.cache_dir),
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                FLORENCE_MODEL_ID,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir),
            )
            self.model.eval()
        except Exception as exc:
            self.model = None
            self.processor = None
            raise ClassifierLoadError(
                f"Could not load the existing Florence-2 model ({FLORENCE_MODEL_ID}). "
                "Check the Transformers version and model download access. "
                f"Details: {exc}"
            ) from exc

    @staticmethod
    def _pixel_shares(crop_bgr: np.ndarray) -> dict[str, float]:
        labels = ("black_smoke", "white_smoke", "grey_smoke", "blue_smoke")
        empty = {label: 0.0 for label in labels}
        if crop_bgr.size == 0:
            return empty

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        low_color = saturation < 60
        soft_blue = (hue > 95) & (hue < 135) & (saturation < 120)
        mask = (low_color | soft_blue).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        usable = (mask > 0) & (saturation <= 110)
        count = int(np.count_nonzero(usable))
        if count < 50:
            return empty

        blue = usable & (hue >= 95) & (hue <= 135) & (saturation > 35)
        black = usable & ~blue & (value < 115)
        white = usable & ~blue & (value > 150) & (saturation < 40)
        grey = usable & ~blue & ~black & ~white
        return {
            "black_smoke": round(float(np.count_nonzero(black)) / count, 4),
            "white_smoke": round(float(np.count_nonzero(white)) / count, 4),
            "grey_smoke": round(float(np.count_nonzero(grey)) / count, 4),
            "blue_smoke": round(float(np.count_nonzero(blue)) / count, 4),
        }

    def _caption(self, crop_bgr: np.ndarray) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Florence-2 is not loaded. Call load() first.")
        rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_crop)
        inputs = self.processor(text=CAPTION_TASK, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        model_dtype = next(self.model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text,
            task=CAPTION_TASK,
            image_size=(image.width, image.height),
        )
        return str(parsed.get(CAPTION_TASK, "")).strip()

    def classify(self, crop_bgr: np.ndarray) -> ColorResult:
        """Classify one smoke crop; heuristic scores are not calibrated probabilities."""
        if not isinstance(crop_bgr, np.ndarray) or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError("Expected a color image as a three-channel OpenCV BGR crop.")
        if crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
            return ColorResult(None, "unavailable", 0.0, "pixel_share_heuristic", None, {}, "Empty crop.")

        shares = self._pixel_shares(crop_bgr)
        top_label = max(shares, key=shares.get)
        top_share = shares[top_label]
        try:
            if self.model is None or self.processor is None:
                self.load()
            caption = self._caption(crop_bgr)
        except Exception as exc:
            raise ClassifierLoadError(
                f"Smoke pixel analysis completed, but Florence-2 captioning failed: {exc}"
            ) from exc

        caption_match = COLOR_PATTERN.search(caption)
        caption_label = COLOR_LABELS.get(caption_match.group(1).lower()) if caption_match else None
        if top_share < self.min_pixel_share:
            return ColorResult(
                None,
                "uncertain",
                top_share,
                "pixel_share_heuristic",
                caption,
                shares,
                "No colour reached the configured pixel-share threshold.",
            )
        if caption_label is not None and caption_label != top_label:
            return ColorResult(
                None,
                "uncertain",
                top_share,
                "pixel_share_heuristic",
                caption,
                shares,
                "Florence-2 caption and pixel-colour heuristic disagree.",
            )
        return ColorResult(
            top_label,
            "classified",
            top_share,
            "pixel_share_heuristic",
            caption,
            shares,
            "Colour selected by pixel-share heuristic; score is not a calibrated probability.",
        )
