"""One SKU, end to end: raw photo -> cutout -> scene -> guardrail -> versioned outputs."""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Callable

from PIL import Image

from .config import Settings
from .cutout import Cutout, CutoutError, remove_background
from .guardrail import GuardrailResult, run_guardrail
from .jobqueue import Job
from .scene import SceneBackend
from .storage import Storage, put_version, sku_prefix, write_status

log = logging.getLogger("catalogsnap")

Segmenter = Callable[[Image.Image], Cutout]


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def process_image(
    image: Image.Image,
    settings: Settings,
    segmenter: Segmenter | None = None,
    backend_factory: Callable[[str], SceneBackend] | None = None,
) -> tuple[Cutout, GuardrailResult]:
    """Pure function core of the pipeline. No storage, no queue."""
    fallbacks = tuple(m.strip() for m in settings.cutout_fallback_models.split(",") if m.strip())
    segment = segmenter or (lambda im: remove_background(im, settings.cutout_model, fallbacks))
    cutout = segment(image)
    return cutout, run_guardrail(cutout, settings, backend_factory)


def process_job(
    job: Job,
    storage: Storage,
    settings: Settings,
    segmenter: Segmenter | None = None,
    backend_factory: Callable[[str], SceneBackend] | None = None,
) -> dict:
    """Run one job and persist the results. Never raises: failures are written to status.json."""
    prefix = sku_prefix(job.batch_id, job.sku)
    status: dict = {"batch_id": job.batch_id, "sku": job.sku, "raw_key": job.raw_key, "state": "processing"}
    write_status(storage, prefix, status)
    started = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(storage.get(job.raw_key)))
        image.load()
        cutout, result = process_image(image, settings, segmenter, backend_factory)
        report = {
            "status": result.status,
            "backend": result.scene.backend,
            "cutout": {
                "model": cutout.model,
                "coverage": round(cutout.coverage, 4),
                "holes_filled": cutout.holes_filled,
                "rotation_deg": cutout.rotation_deg,
            },
            "final": result.report.to_dict(),
            "attempts": result.history,
        }
        status["cutout_key"] = put_version(storage, prefix, "cutout", "png", _png(cutout.rgba), "image/png")
        status["report_key"] = put_version(
            storage, prefix, "report", "json", json.dumps(report, indent=2).encode(), "application/json"
        )
        if result.accepted:
            status["processed_key"] = put_version(
                storage, prefix, "processed", "png", _png(result.scene.image), "image/png"
            )
        else:
            # Keep the rejected candidate for review, but never under processed/.
            status["rejected_key"] = put_version(
                storage, prefix, "rejected", "png", _png(result.scene.image), "image/png"
            )
        status.update(
            state=result.status,
            backend=result.scene.backend,
            attempts=len(result.history),
            cutout_model=cutout.model,
            metrics=result.report.to_dict(),
        )
    except (CutoutError, OSError, KeyError) as exc:
        status.update(state="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a worker must survive any single bad job
        log.exception("job %s/%s crashed", job.batch_id, job.sku)
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    status["seconds"] = round(time.perf_counter() - started, 2)
    write_status(storage, prefix, status)
    return status
