"""Contact-shadow synthesis from a product's alpha matte.

Two layers, the way a studio shot actually looks:
  * a tight, dark occlusion shadow hugging the base of the product
  * a wide, soft ambient shadow falling away from the key light
"""

from __future__ import annotations

import cv2
import numpy as np


def _odd(n: float) -> int:
    n = max(int(n), 1)
    return n if n % 2 else n + 1


def synthesize_shadow(
    alpha: np.ndarray,
    canvas_hw: tuple[int, int],
    origin_xy: tuple[int, int],
    light_dx: float = 0.18,
    squash: float = 0.16,
    contact_opacity: float = 0.55,
    ambient_opacity: float = 0.28,
) -> np.ndarray:
    """Return a float32 shadow map in [0, 1] the size of the canvas (1 = fully dark).

    alpha     : uint8 matte of the product, already at its final size
    origin_xy : top-left corner of the product on the canvas
    light_dx  : horizontal shadow throw as a fraction of product height (+ = right)
    squash    : how tall the ground shadow is relative to the product
    """
    ch, cw = canvas_hw
    ph, pw = alpha.shape
    ox, oy = origin_xy
    base_y = oy + ph  # the ground line the product stands on
    matte = alpha.astype(np.float32) / 255.0

    # Project the silhouette onto the ground plane: flip it, squash it and shear it away from the light.
    src = np.float32([[0, ph], [pw, ph], [0, 0]])
    dst = np.float32(
        [
            [ox, base_y],
            [ox + pw, base_y],
            [ox + light_dx * ph, base_y + squash * ph],
        ]
    )
    cast = cv2.warpAffine(matte, cv2.getAffineTransform(src, dst), (cw, ch), flags=cv2.INTER_LINEAR)

    # Shadows get softer and lighter the further they are from the contact line.
    rows = np.arange(ch, dtype=np.float32)[:, None]
    falloff = np.clip(1.0 - (rows - base_y) / max(squash * ph, 1.0), 0.0, 1.0)
    ambient = cv2.GaussianBlur(cast * falloff, (0, 0), sigmaX=max(pw * 0.035, 2.0))

    # Occlusion: the widest part of the bottom slice of the product, pressed into the ground.
    band = max(int(ph * 0.06), 2)
    footprint = matte[-band:, :].max(axis=0)
    thick = max(int(ph * 0.022), 2)
    contact = np.zeros((ch, cw), np.float32)
    y0, y1 = max(base_y - thick // 2, 0), min(base_y + thick, ch)
    x0, x1 = max(ox, 0), min(ox + pw, cw)
    if y1 > y0 and x1 > x0:
        contact[y0:y1, x0:x1] = footprint[x0 - ox : x1 - ox][None, :]
    contact = cv2.GaussianBlur(contact, (_odd(pw * 0.06), _odd(thick * 2.5)), 0)

    shadow = 1.0 - (1.0 - contact * contact_opacity) * (1.0 - ambient * ambient_opacity)
    return np.clip(shadow, 0.0, 1.0)
