"""Background removal. Produces an RGBA cutout cropped tight to the product."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import logging

import cv2
import numpy as np
from PIL import Image, ImageOps


log = logging.getLogger("catalogsnap.cutout")


@dataclass
class Cutout:
    """A product isolated from its background.

    rgba is cropped to the product's bounding box. bbox is that box in the
    coordinates of the (EXIF-corrected) source image.
    """

    rgba: Image.Image
    bbox: tuple[int, int, int, int]
    coverage: float  # fraction of the source frame the product occupies
    holes_filled: int = 0  # false holes in the mask that were repaired
    rotation_deg: float = 0.0  # tilt that was straightened out
    model: str = ""  # segmentation model that produced the mask

    @property
    def alpha(self) -> np.ndarray:
        return np.asarray(self.rgba.getchannel("A"))


class CutoutError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _session(model: str):
    import onnxruntime as ort
    from rembg.sessions import sessions_class

    try:
        session_class = next(c for c in sessions_class if c.name() == model)
    except StopIteration:
        raise CutoutError(f"Unknown cutout model: {model}") from None
    # BiRefNet's peak allocation with the default arena is several GB and fails on
    # machines with ~6 GB free. Without the arena it runs in far less, at the same speed.
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return session_class(model, opts)


def _is_out_of_memory(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or "alloc" in str(exc).lower()


def remove_background(
    image: Image.Image,
    model: str = "birefnet-general-lite",
    fallbacks: tuple[str, ...] = (),
) -> Cutout:
    """Segment the product with a rembg model (BiRefNet by default).

    BiRefNet needs a few GB of free RAM. If the machine cannot provide it, the lighter
    models in `fallbacks` are tried in order, and the cutout records which one was used.
    """
    import gc

    from rembg import remove

    image = ImageOps.exif_transpose(image).convert("RGB")
    candidates = [model, *(m for m in fallbacks if m != model)]
    for i, name in enumerate(candidates):
        try:
            mask = np.asarray(remove(image, session=_session(name), only_mask=True))
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own exception types
            if not _is_out_of_memory(exc):
                raise
            _session.cache_clear()  # drop the model that did not fit before loading another
            gc.collect()
            if i + 1 == len(candidates):
                raise CutoutError(
                    f"Not enough free memory to run the cutout model ({', '.join(candidates)}). "
                    "Close other applications and try again."
                ) from None
            log.warning("%s ran out of memory; falling back to %s", name, candidates[i + 1])
            continue
        cutout = finish_cutout(np.asarray(image), mask)
        cutout.model = name
        return cutout
    raise CutoutError("No cutout model configured.")


def finish_cutout(rgb: np.ndarray, mask: np.ndarray, straighten: bool = True) -> Cutout:
    """Clean up a raw segmentation mask and build the cutout."""
    mask, filled = repair_holes(rgb, mask)
    rgba = Image.fromarray(np.dstack([rgb, mask]), "RGBA")
    angle = upright_angle(mask) if straighten else 0.0
    if angle:
        rgba = rgba.rotate(angle, resample=Image.BICUBIC, expand=True)
    cutout = crop_to_alpha(rgba)
    cutout.holes_filled = filled
    cutout.rotation_deg = round(angle, 2)
    return cutout


def repair_holes(rgb: np.ndarray, mask: np.ndarray, min_delta_e: float = 12.0) -> tuple[np.ndarray, int]:
    """Fill holes the segmenter wrongly punched inside the product.

    Segmenters sometimes drop printed dots, highlights or pale label areas. A real see-through
    hole (a mug handle) shows the background through it; a false hole does not. So a hole is
    filled only when its color differs clearly from the background around the product.
    """
    binary = (mask > 127).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask, 0
    ring = cv2.dilate(binary, np.ones((61, 61), np.uint8)) - cv2.dilate(binary, np.ones((15, 15), np.uint8))
    if ring.sum() < 100:
        return mask, 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = np.median(lab[ring > 0], axis=0)
    out, filled = mask.copy(), 0
    for contour, info in zip(contours, hierarchy[0]):
        if info[3] < 0 or cv2.contourArea(contour) < 16:  # outer contour, or speck
            continue
        hole = np.zeros_like(binary)
        cv2.drawContours(hole, [contour], -1, 1, cv2.FILLED)
        inside = (hole > 0) & (binary == 0)
        if not inside.any():
            continue
        diff = np.median(lab[inside], axis=0) - background
        diff[0] *= 0.5  # lighting changes brightness across a photo far more than it changes color
        if float(np.linalg.norm(diff)) > min_delta_e:
            cv2.drawContours(out, [contour], -1, 255, cv2.FILLED)
            filled += 1
    return out, filled


def upright_angle(mask: np.ndarray, max_deg: float = 10.0) -> float:
    """Degrees to rotate (counter-clockwise) so a slightly tilted, boxy product stands straight.

    Returns 0 for round or irregular shapes, where a bounding rectangle's angle means nothing,
    and for tilts large enough to be deliberate.
    """
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    (_, _), (w, h), angle = cv2.minAreaRect(contour)
    if w * h == 0 or cv2.contourArea(contour) / (w * h) < 0.72:
        return 0.0
    angle = (angle + 45.0) % 90.0 - 45.0  # tilt of the nearest axis, in (-45, 45]
    return float(angle) if 0.4 < abs(angle) <= max_deg else 0.0


def crop_to_alpha(rgba: Image.Image, threshold: int = 12, pad: int = 2) -> Cutout:
    """Crop an RGBA image to the bounding box of its visible pixels."""
    rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    visible = alpha > threshold
    coverage = float(visible.mean())
    if coverage < 0.005:
        raise CutoutError("No product found: the mask is empty.")
    if coverage > 0.97:
        raise CutoutError("Segmentation kept the whole frame; no background was removed.")
    ys, xs = np.where(visible)
    h, w = alpha.shape
    box = (
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + 1 + pad, w),
        min(int(ys.max()) + 1 + pad, h),
    )
    return Cutout(rgba=rgba.crop(box), bbox=box, coverage=coverage)
