from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from catalogsnap.cutout import finish_cutout, repair_holes, upright_angle


def _scene_with_holes():
    """Gray backdrop, blue product. One hole shows a printed red dot, one shows the backdrop."""
    rgb = np.full((400, 400, 3), (150, 150, 150), np.uint8)
    rgb[80:320, 100:300] = (30, 70, 150)
    rgb[120:160, 130:170] = (200, 40, 40)  # printed dot the segmenter dropped
    rgb[220:280, 180:260] = (150, 150, 150)  # a real see-through hole
    mask = np.zeros((400, 400), np.uint8)
    mask[80:320, 100:300] = 255
    mask[120:160, 130:170] = 0
    mask[220:280, 180:260] = 0
    return rgb, mask


def test_false_holes_are_filled_and_real_ones_kept():
    rgb, mask = _scene_with_holes()
    repaired, filled = repair_holes(rgb, mask)
    assert filled == 1
    assert repaired[140, 150] == 255  # printed dot is part of the product again
    assert repaired[250, 220] == 0  # see-through hole still transparent


@pytest.mark.parametrize("tilt", [-6.0, -2.5, 3.0, 7.0])
def test_tilted_box_is_straightened(tilt):
    base = Image.new("L", (600, 600), 0)
    base.paste(255, (180, 120, 420, 480))
    mask = np.asarray(base.rotate(tilt, resample=Image.BICUBIC))
    rgb = np.full((600, 600, 3), 120, np.uint8)

    cutout = finish_cutout(rgb, mask)

    assert abs(cutout.rotation_deg) == pytest.approx(abs(tilt), abs=0.6)
    assert upright_angle(cutout.alpha) == 0.0  # nothing left to correct
    w, h = cutout.rgba.size
    assert w == pytest.approx(240, abs=8) and h == pytest.approx(360, abs=8)


def test_round_products_are_not_rotated():
    yy, xx = np.mgrid[0:400, 0:400]
    disc = ((xx - 200) ** 2 + (yy - 200) ** 2 < 120**2).astype(np.uint8) * 255
    assert upright_angle(disc) == 0.0


def test_falls_back_to_a_lighter_model_when_out_of_memory(monkeypatch):
    import rembg

    from catalogsnap import cutout as cutout_module
    from catalogsnap.cutout import CutoutError, remove_background

    fake_session = lambda name: name  # noqa: E731
    fake_session.cache_clear = lambda: None
    monkeypatch.setattr(cutout_module, "_session", fake_session)

    def fake_remove(image, session, only_mask):
        if session == "big":
            raise RuntimeError("Status Message: bad allocation")
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (100, 100, 300, 400))
        return mask

    monkeypatch.setattr(rembg, "remove", fake_remove)
    photo = Image.new("RGB", (400, 500), (120, 120, 120))

    result = remove_background(photo, "big", fallbacks=("small",))
    assert result.model == "small"

    with pytest.raises(CutoutError, match="memory"):
        remove_background(photo, "big")
