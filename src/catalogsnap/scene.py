"""Scene backends: turn a product cutout into a finished catalog image.

studio   deterministic compositor. Brand background, synthesized contact shadow, and the
         product pasted untouched, so labels and colors cannot drift.
kontext  FLUX.1 Kontext (image-to-image) through the Black Forest Labs API. It relights the
         product inside a lifestyle scene, which is exactly why its output goes through the
         guardrail before anything is accepted.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image, ImageColor

from .config import Settings
from .cutout import Cutout
from .shadow import synthesize_shadow


@dataclass
class Scene:
    image: Image.Image  # RGB, canvas_size x canvas_size
    reference: Image.Image  # RGBA product at the size it was placed
    box: tuple[int, int, int, int]  # x, y, w, h where the product is expected on the canvas
    backend: str
    meta: dict = field(default_factory=dict)


class SceneBackend(Protocol):
    name: str

    def render(self, cutout: Cutout, attempt: int = 0) -> Scene: ...


class SceneError(RuntimeError):
    pass


def _gradient(size: int, top: str, bottom: str) -> np.ndarray:
    a = np.array(ImageColor.getrgb(top), np.float32)
    b = np.array(ImageColor.getrgb(bottom), np.float32)
    t = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    return np.broadcast_to(a + (b - a) * t, (size, size, 3)).copy()


class StudioBackend:
    name = "studio"

    def __init__(self, settings: Settings):
        self.s = settings

    def render(self, cutout: Cutout, attempt: int = 0) -> Scene:
        size = self.s.canvas_size
        pw, ph = cutout.rgba.size
        scale = self.s.product_fill * size / max(pw, ph)
        w, h = max(round(pw * scale), 1), max(round(ph * scale), 1)
        product = cutout.rgba.resize((w, h), Image.LANCZOS)

        # Every SKU shares one ground line and one centre line. That is the consistency.
        ground = round(size * (0.5 + self.s.product_fill / 2))
        x, y = (size - w) // 2, ground - h

        canvas = _gradient(size, self.s.background_top, self.s.background_bottom)
        alpha = np.asarray(product.getchannel("A"))
        shadow = synthesize_shadow(alpha, (size, size), (x, y))
        canvas *= 1.0 - shadow[..., None]

        image = Image.fromarray(canvas.clip(0, 255).astype(np.uint8), "RGB")
        image.paste(product, (x, y), product)
        return Scene(image=image, reference=product, box=(x, y, w, h), backend=self.name)


class KontextBackend:
    """FLUX.1 Kontext via api.bfl.ai. Each retry uses a new seed and a stricter prompt."""

    name = "kontext"

    def __init__(self, settings: Settings):
        if not settings.bfl_api_key:
            raise SceneError("scene_backend=kontext needs CATALOGSNAP_BFL_API_KEY.")
        self.s = settings
        self._draft = StudioBackend(settings)

    def render(self, cutout: Cutout, attempt: int = 0) -> Scene:
        import httpx

        draft = self._draft.render(cutout)
        buf = io.BytesIO()
        draft.image.save(buf, "PNG")
        prompt = self.s.scene_prompt
        if attempt:
            prompt += (
                " Do not redraw, restyle, recolor, resize or move the product."
                " Every letter and logo on it must stay pixel-identical."
            )
        seed = 1000 + attempt
        headers = {"x-key": self.s.bfl_api_key, "accept": "application/json"}
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"{self.s.bfl_base_url}/{self.s.bfl_model}",
                headers=headers,
                json={
                    "prompt": prompt,
                    "input_image": base64.b64encode(buf.getvalue()).decode(),
                    "seed": seed,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                },
            )
            r.raise_for_status()
            polling_url = r.json()["polling_url"]
            deadline = time.monotonic() + 180
            while True:
                if time.monotonic() > deadline:
                    raise SceneError("Kontext generation timed out.")
                time.sleep(1.5)
                status = client.get(polling_url, headers=headers).json()
                if status.get("status") == "Ready":
                    result = client.get(status["result"]["sample"])
                    result.raise_for_status()
                    break
                if status.get("status") not in ("Pending", "Queued", "Processing"):
                    raise SceneError(f"Kontext generation failed: {status.get('status')}")
        size = self.s.canvas_size
        image = Image.open(io.BytesIO(result.content)).convert("RGB").resize((size, size), Image.LANCZOS)
        return Scene(
            image=image,
            reference=draft.reference,
            box=draft.box,
            backend=self.name,
            meta={"seed": seed, "prompt": prompt},
        )


def make_backend(name: str, settings: Settings) -> SceneBackend:
    if name == "studio":
        return StudioBackend(settings)
    if name == "kontext":
        return KontextBackend(settings)
    raise SceneError(f"Unknown scene backend: {name}")
