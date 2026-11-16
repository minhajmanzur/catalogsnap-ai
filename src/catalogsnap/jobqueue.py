"""Job queue: in-process for local development, Redis when workers run as separate pods."""

from __future__ import annotations

import json
import queue
from dataclasses import asdict, dataclass
from typing import Protocol

from .config import Settings


@dataclass(frozen=True)
class Job:
    batch_id: str
    sku: str
    raw_key: str

    def dumps(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def loads(cls, raw: str | bytes) -> "Job":
        return cls(**json.loads(raw))


class JobQueue(Protocol):
    def push(self, job: Job) -> None: ...
    def pop(self, timeout: float = 5.0) -> Job | None: ...
    def ack(self, job: Job) -> None: ...
    def depth(self) -> int: ...


class MemoryQueue:
    def __init__(self) -> None:
        self._q: queue.Queue[Job] = queue.Queue()

    def push(self, job: Job) -> None:
        self._q.put(job)

    def pop(self, timeout: float = 5.0) -> Job | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def ack(self, job: Job) -> None:
        pass

    def depth(self) -> int:
        return self._q.qsize()


class RedisQueue:
    """Reliable queue: a popped job sits in a processing list until it is acked.

    If a worker pod is killed mid-job (scale-down, eviction), the job is still in
    `{name}:processing` and `requeue_stale()` puts it back.
    """

    def __init__(self, client, name: str):
        self.r = client
        self.name = name
        self.processing = f"{name}:processing"

    def push(self, job: Job) -> None:
        self.r.lpush(self.name, job.dumps())

    def pop(self, timeout: float = 5.0) -> Job | None:
        raw = self.r.blmove(self.name, self.processing, timeout, "RIGHT", "LEFT")
        return Job.loads(raw) if raw else None

    def ack(self, job: Job) -> None:
        self.r.lrem(self.processing, 1, job.dumps())

    def depth(self) -> int:
        return int(self.r.llen(self.name))

    def requeue_stale(self) -> int:
        moved = 0
        while self.r.lmove(self.processing, self.name, "RIGHT", "LEFT"):
            moved += 1
        return moved


def make_queue(settings: Settings) -> JobQueue:
    if settings.queue_backend == "redis":
        import redis

        return RedisQueue(redis.Redis.from_url(settings.redis_url, decode_responses=True), settings.queue_name)
    return MemoryQueue()
