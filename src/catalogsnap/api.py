"""FastAPI service: upload a batch, poll it, fetch results."""

from __future__ import annotations

import io
import json
import threading
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import Settings, get_settings
from .jobqueue import Job, JobQueue, MemoryQueue, make_queue
from .storage import (
    Storage,
    latest_version,
    make_storage,
    put_version,
    read_status,
    safe_name,
    sku_prefix,
    write_status,
)
from .worker import run_worker

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
FORMATS = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"), "WEBP": ("webp", "image/webp")}


def create_app(
    settings: Settings | None = None,
    storage: Storage | None = None,
    queue: JobQueue | None = None,
    **pipeline_kwargs,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.storage = storage or make_storage(settings)
        app.state.queue = queue or make_queue(settings)
        stop = threading.Event()
        if isinstance(app.state.queue, MemoryQueue):
            # Single-process dev mode: no Redis, so the API runs the worker itself.
            threading.Thread(
                target=run_worker,
                args=(app.state.queue, app.state.storage, settings, stop),
                kwargs=pipeline_kwargs,
                daemon=True,
            ).start()
        yield
        stop.set()

    app = FastAPI(title="CatalogSnap AI", version="0.1.0", lifespan=lifespan)

    def _name(value: str) -> str:
        try:
            return safe_name(value)
        except ValueError:
            raise HTTPException(400, f"Invalid name: {value!r}") from None

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "scene_backend": settings.scene_backend, "queue_depth": app.state.queue.depth()}

    @app.post("/batches", status_code=202)
    async def create_batch(files: list[UploadFile] = File(...)) -> dict:
        """Upload raw product photos. The file name (without extension) becomes the SKU."""
        batch_id = uuid.uuid4().hex[:12]
        accepted, skipped = [], []
        for upload in files:
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            name = upload.filename or "unnamed"
            if len(data) > MAX_UPLOAD_BYTES:
                skipped.append({"file": name, "reason": "larger than 25 MB"})
                continue
            try:
                with Image.open(io.BytesIO(data)) as probe:
                    fmt = probe.format
                    probe.verify()
                ext, content_type = FORMATS[fmt]
                sku = safe_name(PurePath(name).stem)
            except (UnidentifiedImageError, KeyError, ValueError, OSError):
                skipped.append({"file": name, "reason": "not a JPEG, PNG or WebP image"})
                continue
            prefix = sku_prefix(batch_id, sku)
            raw_key = put_version(app.state.storage, prefix, "raw", ext, data, content_type)
            write_status(
                app.state.storage,
                prefix,
                {"batch_id": batch_id, "sku": sku, "raw_key": raw_key, "state": "queued"},
            )
            app.state.queue.push(Job(batch_id, sku, raw_key))
            accepted.append(sku)
        if not accepted:
            raise HTTPException(400, {"message": "No usable images in upload.", "skipped": skipped})
        return {"batch_id": batch_id, "queued": accepted, "skipped": skipped}

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict:
        root = f"batches/{_name(batch_id)}"
        items = [
            json.loads(app.state.storage.get(key))
            for key in app.state.storage.list(root)
            if key.endswith("/status.json")
        ]
        if not items:
            raise HTTPException(404, "Unknown batch.")
        counts = Counter(item["state"] for item in items)
        done = all(item["state"] in ("accepted", "rejected", "failed") for item in items)
        return {"batch_id": batch_id, "done": done, "counts": counts, "items": items}

    def _fetch(batch_id: str, sku: str, kind: str, media_type: str) -> Response:
        key = latest_version(app.state.storage, sku_prefix(_name(batch_id), _name(sku)), kind)
        if key is None:
            status = read_status(app.state.storage, sku_prefix(_name(batch_id), _name(sku)))
            if status is None:
                raise HTTPException(404, "Unknown batch or SKU.")
            raise HTTPException(409, f"No {kind} output; job state is '{status['state']}'.")
        return Response(app.state.storage.get(key), media_type=media_type, headers={"X-Asset-Key": key})

    @app.get("/batches/{batch_id}/{sku}/image")
    def get_image(batch_id: str, sku: str) -> Response:
        """Latest accepted catalog image for a SKU."""
        return _fetch(batch_id, sku, "processed", "image/png")

    @app.get("/batches/{batch_id}/{sku}/cutout")
    def get_cutout(batch_id: str, sku: str) -> Response:
        return _fetch(batch_id, sku, "cutout", "image/png")

    @app.get("/batches/{batch_id}/{sku}/report")
    def get_report(batch_id: str, sku: str) -> Response:
        """Guardrail metrics for every attempt made on this SKU."""
        return _fetch(batch_id, sku, "report", "application/json")

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn catalogsnap.api:app_factory --factory`."""
    return create_app()
