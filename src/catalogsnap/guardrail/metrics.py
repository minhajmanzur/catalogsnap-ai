"""Product-fidelity checks: did the generated image keep the product intact?

The reference is the product cutout at the size it was placed. The candidate is the finished
scene. We first register the candidate back onto the reference (SIFT + RANSAC homography), then
measure, inside the product mask only:

  ssim          structural similarity on luminance  -> catches redrawn labels and smeared text
  inlier_ratio  share of feature matches that agree on one rigid placement -> catches warping
  anisotropy    stretch hidden in the homography    -> catches squashed or widened products
  delta_e       median CIEDE2000 color difference   -> catches color drift
  hue_shift     circular mean hue difference        -> catches brand colors changing hue
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

MIN_MATCHES = 12


@dataclass
class Thresholds:
    min_ssim: float = 0.80
    min_inlier_ratio: float = 0.45
    max_delta_e: float = 9.0
    max_hue_shift_deg: float = 12.0
    max_anisotropy: float = 1.08


@dataclass
class FidelityReport:
    passed: bool
    ssim: float
    delta_e: float
    hue_shift_deg: float
    inlier_ratio: float | None  # None when the product has too little texture to match
    anisotropy: float | None
    registered: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _register(ref_gray: np.ndarray, mask: np.ndarray, cand_gray: np.ndarray, box) -> tuple:
    """Find the homography mapping reference coords -> candidate coords."""
    x, y, w, h = box
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_r, des_r = sift.detectAndCompute(ref_gray, mask)
    kp_c, des_c = sift.detectAndCompute(cand_gray, None)
    if des_r is None or des_c is None or len(kp_r) < MIN_MATCHES or len(kp_c) < MIN_MATCHES:
        return None, None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_c, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_MATCHES:
        # Texture exists but nothing in the candidate matches it: the product was redrawn.
        return None, 0.0
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_c[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0.0
    return H, float(inliers.sum()) / len(good)


def _anisotropy(H: np.ndarray, w: int, h: int) -> float:
    """Ratio of the two stretch factors of the mapping at the product centre (1.0 = no distortion)."""
    cx, cy = w / 2, h / 2
    pts = np.float32([[[cx, cy]], [[cx + 1, cy]], [[cx, cy + 1]]])
    p = cv2.perspectiveTransform(pts, H)[:, 0]
    J = np.stack([p[1] - p[0], p[2] - p[0]], axis=1)
    s = np.linalg.svd(J, compute_uv=False)
    return float(s[0] / max(s[1], 1e-6))


def check_fidelity(
    reference: Image.Image,
    candidate: Image.Image,
    box: tuple[int, int, int, int],
    thresholds: Thresholds | None = None,
) -> FidelityReport:
    t = thresholds or Thresholds()
    x, y, w, h = box
    ref_rgba = np.asarray(reference.convert("RGBA").resize((w, h), Image.LANCZOS))
    ref_rgb = ref_rgba[..., :3]
    # Erode the mask so soft edges and shadow bleed do not count against the product.
    k = max(int(min(w, h) * 0.015), 1) * 2 + 1
    mask = cv2.erode((ref_rgba[..., 3] > 200).astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    inside = mask > 0
    cand = np.asarray(candidate.convert("RGB"))

    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    cand_gray = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)
    H, inlier_ratio = _register(ref_gray, mask, cand_gray, box)

    failures: list[str] = []
    anisotropy = None
    if H is not None:
        anisotropy = _anisotropy(H, w, h)
        aligned = cv2.warpPerspective(cand, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    else:
        # Fall back to where the product was supposed to be placed.
        aligned = np.zeros_like(ref_rgb)
        crop = cand[max(y, 0) : y + h, max(x, 0) : x + w]
        aligned[: crop.shape[0], : crop.shape[1]] = crop

    if not inside.any():
        return FidelityReport(False, 0.0, 0.0, 0.0, inlier_ratio, anisotropy, H is not None, ["empty product mask"])

    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    _, ssim_map = structural_similarity(ref_gray, aligned_gray, data_range=255, full=True)
    ssim = float(ssim_map[inside].mean())

    # Blur slightly before comparing color so sub-pixel misregistration is not read as drift.
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5)  # noqa: E731
    lab_r, lab_c = rgb2lab(blur(ref_rgb)), rgb2lab(blur(aligned))
    delta_e = float(np.median(deltaE_ciede2000(lab_r, lab_c)[inside]))

    hsv_r = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)[inside].astype(np.float32)
    hsv_c = cv2.cvtColor(aligned, cv2.COLOR_RGB2HSV)[inside].astype(np.float32)
    colored = (hsv_r[:, 1] > 60) & (hsv_r[:, 2] > 40)  # hue is meaningless on grays
    hue_shift = 0.0
    if colored.sum() > 50:
        d = np.deg2rad((hsv_c[colored, 0] - hsv_r[colored, 0]) * 2.0)  # OpenCV hue is 0..180
        hue_shift = float(abs(np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))))

    if ssim < t.min_ssim:
        failures.append(f"structure changed (SSIM {ssim:.2f} < {t.min_ssim})")
    if inlier_ratio is not None and inlier_ratio < t.min_inlier_ratio:
        failures.append(f"label/logo features no longer match ({inlier_ratio:.0%} inliers)")
    if anisotropy is not None and anisotropy > t.max_anisotropy:
        failures.append(f"product is stretched (anisotropy {anisotropy:.2f})")
    if delta_e > t.max_delta_e:
        failures.append(f"colors drifted (median dE {delta_e:.1f} > {t.max_delta_e})")
    if hue_shift > t.max_hue_shift_deg:
        failures.append(f"brand hue shifted {hue_shift:.0f} deg")

    return FidelityReport(
        passed=not failures,
        ssim=round(ssim, 4),
        delta_e=round(delta_e, 2),
        hue_shift_deg=round(hue_shift, 2),
        inlier_ratio=None if inlier_ratio is None else round(inlier_ratio, 3),
        anisotropy=None if anisotropy is None else round(anisotropy, 3),
        registered=H is not None,
        failures=failures,
    )
