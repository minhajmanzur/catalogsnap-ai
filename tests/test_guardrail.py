from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from catalogsnap.guardrail import check_fidelity, run_guardrail
from catalogsnap.scene import Scene, StudioBackend


def _studio(cutout, settings) -> Scene:
    return StudioBackend(settings).render(cutout)


def _replace_product(scene: Scene, new_product: Image.Image) -> Image.Image:
    """Paste a tampered product over the scene, where the original was."""
    x, y, w, h = scene.box
    out = scene.image.copy()
    out.paste(new_product.resize((w, h)), (x, y), scene.reference)
    return out


def test_studio_output_passes(cutout, settings):
    scene = _studio(cutout, settings)
    report = check_fidelity(scene.reference, scene.image, scene.box)
    assert report.passed, report.failures
    assert report.ssim > 0.97
    assert report.delta_e < 1.0


def test_mild_relighting_is_tolerated(cutout, settings):
    scene = _studio(cutout, settings)
    arr = np.asarray(scene.image, np.float32) * 1.06 + 3  # a touch brighter, as a relight would do
    relit = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    report = check_fidelity(scene.reference, relit, scene.box)
    assert report.passed, report.failures


def test_hue_drift_is_caught(cutout, settings):
    scene = _studio(cutout, settings)
    hsv = cv2.cvtColor(np.asarray(scene.reference.convert("RGB")), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + 20) % 180  # brand blue -> purple
    drifted = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    report = check_fidelity(scene.reference, _replace_product(scene, drifted), scene.box)
    assert not report.passed
    assert any("hue" in f or "color" in f for f in report.failures)


def test_smeared_label_is_caught(cutout, settings):
    scene = _studio(cutout, settings)
    smeared = scene.reference.convert("RGB").filter(ImageFilter.GaussianBlur(7))
    report = check_fidelity(scene.reference, _replace_product(scene, smeared), scene.box)
    assert not report.passed
    assert any("structure" in f or "features" in f for f in report.failures)


def test_stretched_product_is_caught(cutout, settings):
    scene = _studio(cutout, settings)
    x, y, w, h = scene.box
    wide = scene.reference.resize((int(w * 1.25), h))
    out = Image.new("RGB", scene.image.size, (240, 238, 232))
    out.paste(wide, (x - int(w * 0.125), y), wide)
    report = check_fidelity(scene.reference, out, scene.box)
    assert not report.passed
    assert report.anisotropy is not None and report.anisotropy > 1.15


class DriftingBackend:
    """Imitates a generative model that recolors the product on every attempt."""

    name = "kontext"

    def __init__(self, settings):
        self.studio = StudioBackend(settings)
        self.calls = 0

    def render(self, cutout, attempt=0):
        self.calls += 1
        scene = self.studio.render(cutout)
        hsv = cv2.cvtColor(np.asarray(scene.image), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 25) % 180
        scene.image = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        scene.backend = self.name
        return scene


def test_agent_retries_then_falls_back_to_studio(cutout, settings):
    s = settings.model_copy(update={"scene_backend": "kontext", "max_attempts": 3})
    drifting = DriftingBackend(s)
    factory = lambda name: drifting if name == "kontext" else StudioBackend(s)  # noqa: E731

    result = run_guardrail(cutout, s, factory)

    assert result.accepted
    assert result.scene.backend == "studio"
    assert drifting.calls == 3
    assert [h["backend"] for h in result.history] == ["kontext"] * 3 + ["studio"]
    assert [h["passed"] for h in result.history] == [False, False, False, True]


def test_agent_rejects_when_fallback_disabled(cutout, settings):
    s = settings.model_copy(update={"scene_backend": "kontext", "max_attempts": 2, "fallback_to_studio": False})
    drifting = DriftingBackend(s)

    result = run_guardrail(cutout, s, lambda name: drifting)

    assert result.status == "rejected"
    assert drifting.calls == 2
    assert result.report.failures
