"""Draw a few fake "phone photos" of products so the pipeline can be tried without real data.

Each has a textured, unevenly lit background, a slightly rotated product with a printed label,
and sensor noise. They are stand-ins: real photos are a harder test for the cutout model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _background(rng: np.random.Generator, w: int, h: int, tone: tuple[int, int, int]) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1.15 - 0.55 * np.hypot((xx - w * 0.25) / w, (yy - h * 0.2) / h)  # lamp at top-left
    grain = np.sin(yy / 9.0 + rng.normal(0, 0.6, (h, 1))) * 6 + rng.normal(0, 5, (h, w))
    img = np.array(tone, np.float32)[None, None] * light[..., None] + grain[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(1.2))


def _bottle(body: tuple, cap: tuple, label: tuple, brand: str, sub: str) -> Image.Image:
    im = Image.new("RGBA", (420, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((150, 10, 270, 120), 18, fill=cap)
    d.rounded_rectangle((165, 110, 255, 200), 10, fill=body)
    d.rounded_rectangle((40, 180, 380, 890), 70, fill=body)
    d.rounded_rectangle((60, 380, 360, 760), 16, fill=label)
    d.text((210, 470), brand, font=_font(64), fill=body, anchor="mm")
    d.line((100, 530, 320, 530), fill=body, width=5)
    d.text((210, 590), sub, font=_font(30), fill=(40, 40, 40), anchor="mm")
    d.ellipse((170, 640, 250, 720), outline=body, width=8)
    d.rounded_rectangle((70, 220, 110, 840), 20, fill=(255, 255, 255, 60))  # specular streak
    return im


def _box(body: tuple, accent: tuple, brand: str, sub: str) -> Image.Image:
    im = Image.new("RGBA", (620, 760), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 10, 610, 750), 14, fill=body)
    d.rectangle((10, 520, 610, 750), fill=accent)
    d.text((310, 200), brand, font=_font(92), fill=accent, anchor="mm")
    d.text((310, 330), sub, font=_font(40), fill=(30, 30, 30), anchor="mm")
    d.text((310, 635), "NET WT 250 g", font=_font(44), fill=body, anchor="mm")
    for i in range(6):
        d.ellipse((70 + i * 85, 410, 120 + i * 85, 460), fill=accent)
    return im


def _photo(rng: np.random.Generator, product: Image.Image, tone: tuple, angle: float) -> Image.Image:
    w, h = 1200, 1500
    bg = _background(rng, w, h, tone)
    product = product.rotate(angle, expand=True, resample=Image.BICUBIC)
    x, y = (w - product.width) // 2 + 40, h - product.height - 170
    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).ellipse(
        (x - 20, y + product.height - 50, x + product.width + 130, y + product.height + 45), fill=150
    )
    bg = Image.composite(Image.new("RGB", (w, h), (20, 18, 15)), bg, shadow.filter(ImageFilter.GaussianBlur(28)))
    bg.paste(product, (x, y), product)
    noisy = np.asarray(bg, np.float32) + rng.normal(0, 4, (h, w, 3))
    return Image.fromarray(noisy.clip(0, 255).astype(np.uint8), "RGB")


def main(out_dir: str = "samples/raw") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    items = {
        "SKU-1001-serum": (_bottle((24, 92, 84), (30, 30, 30), (245, 240, 228), "VERDA", "hydrating serum"), (176, 150, 120), 3),
        "SKU-1002-tonic": (_bottle((178, 54, 44), (235, 235, 235), (250, 246, 238), "ROSSO", "daily tonic 200 ml"), (120, 126, 134), -4),
        "SKU-2001-tea": (_box((244, 236, 214), (32, 70, 140), "AZUL", "organic black tea"), (150, 132, 108), 2),
        "SKU-2002-oats": (_box((250, 222, 120), (120, 60, 20), "MORN", "rolled oats"), (98, 110, 100), -3),
    }
    for sku, (product, tone, angle) in items.items():
        _photo(rng, product, tone, angle).save(out / f"{sku}.jpg", quality=90)
        print("wrote", out / f"{sku}.jpg")


if __name__ == "__main__":
    main(*sys.argv[1:])
