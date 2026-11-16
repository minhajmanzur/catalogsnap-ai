"""Versioned asset storage: local filesystem for development, S3 or MinIO in a cluster.

Key layout (identical on both backends):

    batches/{batch_id}/{sku}/raw/v0001.jpg
    batches/{batch_id}/{sku}/processed/v0001.png
    batches/{batch_id}/{sku}/report/v0001.json
    batches/{batch_id}/{sku}/status.json          <- the only key that is overwritten

Versions are explicit in the key, so history works on any S3-compatible store without
depending on bucket versioning being switched on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from .config import Settings

_VERSION = re.compile(r"/v(\d{4})\.[^/]+$")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """Make a SKU or batch id safe to use as one path segment."""
    cleaned = _SAFE.sub("-", name).strip(".-")
    if not cleaned:
        raise ValueError(f"Unusable name: {name!r}")
    return cleaned[:120]


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Key escapes the storage root: {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        )


class S3Storage:
    def __init__(self, settings: Settings):
        import boto3

        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )

    def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def list(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        keys: list[str] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)


def make_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        storage = S3Storage(settings)
        storage.ensure_bucket()
        return storage
    return LocalStorage(settings.local_storage_dir)


# ---- key helpers -------------------------------------------------------------------------


def sku_prefix(batch_id: str, sku: str) -> str:
    return f"batches/{safe_name(batch_id)}/{safe_name(sku)}"


def put_version(storage: Storage, prefix: str, kind: str, ext: str, data: bytes, content_type: str) -> str:
    """Store data as the next version under {prefix}/{kind}/ and return its key."""
    versions = [int(m.group(1)) for k in storage.list(f"{prefix}/{kind}") if (m := _VERSION.search(k))]
    key = f"{prefix}/{kind}/v{max(versions, default=0) + 1:04d}.{ext.lstrip('.')}"
    storage.put(key, data, content_type)
    return key


def latest_version(storage: Storage, prefix: str, kind: str) -> str | None:
    keys = [k for k in storage.list(f"{prefix}/{kind}") if _VERSION.search(k)]
    return keys[-1] if keys else None


def write_status(storage: Storage, prefix: str, status: dict) -> None:
    storage.put(f"{prefix}/status.json", json.dumps(status, indent=2).encode(), "application/json")


def read_status(storage: Storage, prefix: str) -> dict | None:
    try:
        return json.loads(storage.get(f"{prefix}/status.json"))
    except KeyError:
        return None
