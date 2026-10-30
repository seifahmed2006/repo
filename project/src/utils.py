"""Shared helpers: config loading, seeding, paths."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def out_dir(*parts: str) -> Path:
    p = ROOT.joinpath("outputs", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_features(cfg: dict, task: str) -> list[str]:
    t = cfg["tasks"][task]
    return list(t["numeric"]) + list(t["categorical"])
