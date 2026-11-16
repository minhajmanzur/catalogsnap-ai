"""Runtime configuration, read from environment variables (prefix CATALOGSNAP_)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CATALOGSNAP_", env_file=".env", extra="ignore")

    # --- cutout ---
    # Any rembg session name: birefnet-general, birefnet-general-lite, u2net, isnet-general-use, sam ...
    cutout_model: str = "birefnet-general-lite"
    # Tried in order when the main model does not fit in free RAM. Comma-separated; empty disables.
    cutout_fallback_models: str = "isnet-general-use,u2netp"

    # --- scene generation ---
    # studio  : deterministic compositor (brand background + synthesized shadow). Product pixels are untouched.
    # kontext : FLUX.1 Kontext over the Black Forest Labs HTTP API. Needs CATALOGSNAP_BFL_API_KEY.
    scene_backend: Literal["studio", "kontext"] = "studio"
    bfl_api_key: str | None = None
    bfl_base_url: str = "https://api.bfl.ai/v1"
    bfl_model: str = "flux-kontext-pro"

    # --- output canvas / brand look ---
    canvas_size: int = 1600
    product_fill: float = Field(0.72, description="Longest product side as a fraction of the canvas")
    background_top: str = "#f6f4f0"
    background_bottom: str = "#e7e3db"
    scene_prompt: str = (
        "Place this exact product on a light oak table in a bright, softly lit kitchen. "
        "Keep the product, its label, logo, text and colors exactly unchanged. "
        "Add a natural contact shadow that matches the scene lighting."
    )

    # --- guardrail thresholds ---
    min_ssim: float = 0.80
    min_inlier_ratio: float = 0.45
    max_delta_e: float = 9.0
    max_hue_shift_deg: float = 12.0
    max_attempts: int = 3
    fallback_to_studio: bool = True

    # --- storage ---
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: Path = Path("data/store")
    s3_endpoint_url: str | None = None  # e.g. http://minio:9000 ; leave unset for AWS S3
    s3_bucket: str = "catalogsnap"
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # --- queue ---
    queue_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "catalogsnap:jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
