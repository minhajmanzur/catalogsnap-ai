"""Batch worker. One process per pod; scale the pod count, not threads."""

from __future__ import annotations

import logging
import signal
import threading

from .config import Settings, get_settings
from .jobqueue import JobQueue, make_queue
from .pipeline import process_job
from .storage import Storage, make_storage

log = logging.getLogger("catalogsnap.worker")


def run_worker(
    queue: JobQueue,
    storage: Storage,
    settings: Settings,
    stop: threading.Event | None = None,
    **pipeline_kwargs,
) -> None:
    stop = stop or threading.Event()
    while not stop.is_set():
        job = queue.pop(timeout=2.0)
        if job is None:
            continue
        status = process_job(job, storage, settings, **pipeline_kwargs)
        queue.ack(job)
        log.info("%s/%s -> %s in %ss", job.batch_id, job.sku, status["state"], status["seconds"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    queue = make_queue(settings)
    if hasattr(queue, "requeue_stale") and (n := queue.requeue_stale()):
        log.info("requeued %d job(s) left over from a previous worker", n)
    stop = threading.Event()
    # Kubernetes sends SIGTERM on scale-down: finish the current job, then exit.
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("worker up (scene=%s, cutout=%s)", settings.scene_backend, settings.cutout_model)
    run_worker(queue, make_storage(settings), settings, stop)


if __name__ == "__main__":
    main()
