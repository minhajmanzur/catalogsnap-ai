from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from catalogsnap.config import Settings
from catalogsnap.cutout import Cutout, crop_to_alpha


def _font(size: int):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def make_product() -> Image.Image:
    """A textured RGBA 'box' with a logo and small print, on a transparent margin."""
    im = Image.new("RGBA", (520, 660), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 30, 490, 630), 14, fill=(244, 236, 214, 255))
    d.rectangle((30, 440, 490, 630), fill=(32, 70, 140, 255))
    d.text((260, 170), "AZUL", font=_font(84), fill=(32, 70, 140, 255), anchor="mm")
    d.text((260, 280), "organic black tea", font=_font(34), fill=(30, 30, 30, 255), anchor="mm")
    d.text((260, 535), "NET WT 250 g", font=_font(40), fill=(244, 236, 214, 255), anchor="mm")
    for i in range(5):
        d.ellipse((70 + i * 85, 350, 115 + i * 85, 395), fill=(178, 54, 44, 255))
    return im


@pytest.fixture
def product() -> Image.Image:
    return make_product()


@pytest.fixture
def cutout(product) -> Cutout:
    return crop_to_alpha(product)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, canvas_size=800, local_storage_dir=tmp_path / "store")


@pytest.fixture
def fake_segmenter():
    """Stands in for BiRefNet so tests need no model download: the 'photo' is the product on gray."""

    def segment(image: Image.Image) -> Cutout:
        return crop_to_alpha(make_product())

    return segment


@pytest.fixture
def raw_photo_bytes() -> bytes:
    photo = Image.new("RGB", (640, 800), (150, 140, 120))
    product = make_product()
    photo.paste(product, (60, 70), product)
    buf = io.BytesIO()
    photo.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))
