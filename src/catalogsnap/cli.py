"""Command line: process a folder locally, or start the API / a worker."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from .config import get_settings

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _run(args: argparse.Namespace) -> int:
    from .jobqueue import Job
    from .pipeline import process_job
    from .storage import LocalStorage, put_version, sku_prefix

    settings = get_settings()
    if args.backend:
        settings = settings.model_copy(update={"scene_backend": args.backend})
    files = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 1
    out = Path(args.out)
    storage = LocalStorage(out / "store")
    catalog = out / "catalog"
    catalog.mkdir(parents=True, exist_ok=True)
    batch_id = args.batch or uuid.uuid4().hex[:12]

    print(f"batch {batch_id}: {len(files)} image(s), scene backend = {settings.scene_backend}")
    print(f"{'sku':<28}{'state':<10}{'backend':<9}{'tries':<6}{'ssim':<7}{'dE':<6}{'sec':<7}cutout model")
    bad = 0
    for path in files:
        prefix = sku_prefix(batch_id, path.stem)
        ext = path.suffix.lower().lstrip(".")
        raw_key = put_version(storage, prefix, "raw", ext, path.read_bytes(), "application/octet-stream")
        status = process_job(Job(batch_id, path.stem, raw_key), storage, settings)
        m = status.get("metrics", {})
        print(
            f"{path.stem[:27]:<28}{status['state']:<10}{status.get('backend', '-'):<9}"
            f"{status.get('attempts', '-')!s:<6}{m.get('ssim', '-')!s:<7}{m.get('delta_e', '-')!s:<6}"
            f"{status['seconds']!s:<7}{status.get('cutout_model', '')}"
            + (f" {status['error']}" if "error" in status else "")
        )
        if status["state"] == "accepted":
            (catalog / f"{path.stem}.png").write_bytes(storage.get(status["processed_key"]))
        else:
            bad += 1
    print(f"\ncatalog images: {catalog.resolve()}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catalogsnap", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process every image in a folder, locally")
    run.add_argument("input", help="folder of raw product photos")
    run.add_argument("--out", default="output", help="where to write results (default: ./output)")
    run.add_argument("--batch", help="batch id (default: random)")
    run.add_argument("--backend", choices=["studio", "kontext"], help="override the scene backend")

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("worker", help="start a queue worker")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.command == "run":
        return _run(args)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("catalogsnap.api:app_factory", factory=True, host=args.host, port=args.port)
        return 0
    from .worker import main as worker_main

    worker_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
