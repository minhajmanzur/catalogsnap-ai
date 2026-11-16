from __future__ import annotations

import io
import time

import boto3
import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from PIL import Image

from catalogsnap.api import create_app
from catalogsnap.cutout import CutoutError, crop_to_alpha
from catalogsnap.jobqueue import Job, MemoryQueue, RedisQueue
from catalogsnap.shadow import synthesize_shadow
from catalogsnap.storage import LocalStorage, S3Storage, latest_version, put_version, safe_name


def test_shadow_sits_under_the_product(cutout):
    alpha = cutout.alpha
    h, w = alpha.shape
    shadow = synthesize_shadow(alpha, (1000, 1000), (200, 100))
    assert 0.0 <= shadow.min() and shadow.max() <= 1.0
    base = 100 + h
    assert shadow[base - 2 : base + 6, 200 : 200 + w].mean() > 0.2  # dark at the contact line
    assert shadow[:50].max() < 0.01  # nothing above the product
    assert shadow[base + 5 :, :].mean() > shadow[: base - h // 2, :].mean()


def test_cutout_rejects_empty_and_full_masks():
    with pytest.raises(CutoutError):
        crop_to_alpha(Image.new("RGBA", (50, 50), (0, 0, 0, 0)))
    with pytest.raises(CutoutError):
        crop_to_alpha(Image.new("RGBA", (50, 50), (9, 9, 9, 255)))


def _versioning_roundtrip(storage):
    k1 = put_version(storage, "batches/b/sku", "processed", "png", b"one", "image/png")
    k2 = put_version(storage, "batches/b/sku", "processed", "png", b"two", "image/png")
    assert k1.endswith("processed/v0001.png") and k2.endswith("processed/v0002.png")
    assert latest_version(storage, "batches/b/sku", "processed") == k2
    assert storage.get(k1) == b"one"  # old versions are never overwritten
    assert storage.exists(k2) and not storage.exists("batches/b/sku/processed/v0003.png")
    with pytest.raises(KeyError):
        storage.get("batches/b/missing")


def test_local_storage_versions(tmp_path):
    storage = LocalStorage(tmp_path)
    _versioning_roundtrip(storage)
    with pytest.raises(ValueError):
        storage.put("../escape.txt", b"x")


def test_s3_storage_versions(settings):
    with mock_aws():
        s = settings.model_copy(update={"s3_access_key": "test", "s3_secret_key": "test"})
        storage = S3Storage(s)
        storage.ensure_bucket()
        _versioning_roundtrip(storage)
        assert boto3.client("s3", region_name="us-east-1").list_buckets()["Buckets"][0]["Name"] == "catalogsnap"


def test_safe_name():
    assert safe_name("SKU 1001/red") == "SKU-1001-red"
    assert safe_name("../../etc") == "etc"
    with pytest.raises(ValueError):
        safe_name("..")


def test_redis_queue_survives_a_dead_worker():
    q = RedisQueue(fakeredis.FakeRedis(decode_responses=True), "jobs")
    a, b = Job("b1", "sku-a", "k/a"), Job("b1", "sku-b", "k/b")
    q.push(a)
    q.push(b)
    assert q.depth() == 2
    first = q.pop(timeout=1)
    assert first == a  # FIFO
    q.ack(first)
    second = q.pop(timeout=1)  # this worker "dies" before acking
    assert second == b and q.depth() == 0
    assert q.requeue_stale() == 1
    assert q.pop(timeout=1) == b


def _wait_done(client, batch_id, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        body = client.get(f"/batches/{batch_id}").json()
        if body["done"]:
            return body
        time.sleep(0.1)
    raise AssertionError("batch did not finish")


def test_api_end_to_end(settings, fake_segmenter, raw_photo_bytes):
    app = create_app(settings, LocalStorage(settings.local_storage_dir), MemoryQueue(), segmenter=fake_segmenter)
    with TestClient(app) as client:
        r = client.post(
            "/batches",
            files=[
                ("files", ("SKU-1.jpg", raw_photo_bytes, "image/jpeg")),
                ("files", ("notes.txt", b"not an image", "text/plain")),
            ],
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["queued"] == ["SKU-1"]
        assert body["skipped"][0]["file"] == "notes.txt"

        batch = _wait_done(client, body["batch_id"])
        item = batch["items"][0]
        assert item["state"] == "accepted", item
        assert item["metrics"]["passed"] is True

        img = client.get(f"/batches/{body['batch_id']}/SKU-1/image")
        assert img.status_code == 200
        assert img.headers["x-asset-key"].endswith("processed/v0001.png")
        out = Image.open(io.BytesIO(img.content))
        assert out.size == (settings.canvas_size, settings.canvas_size)
        assert np.asarray(out)[5, 5].min() > 200  # brand background, not the phone photo's

        report = client.get(f"/batches/{body['batch_id']}/SKU-1/report").json()
        assert report["status"] == "accepted" and len(report["attempts"]) == 1

        assert client.get("/batches/nope").status_code == 404
        assert client.get(f"/batches/{body['batch_id']}/other/image").status_code == 404


def test_api_rejects_upload_with_no_images(settings):
    app = create_app(settings, LocalStorage(settings.local_storage_dir), MemoryQueue())
    with TestClient(app) as client:
        r = client.post("/batches", files=[("files", ("a.txt", b"hello", "text/plain"))])
        assert r.status_code == 400
