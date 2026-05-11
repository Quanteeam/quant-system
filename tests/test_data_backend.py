"""Smoke tests for data backend selection."""
from __future__ import annotations

import os
import subprocess
import sys


def _backend_module(value: str | None) -> str:
    env = os.environ.copy()
    if value is None:
        env.pop("QUANT_DATA_BACKEND", None)
    else:
        env["QUANT_DATA_BACKEND"] = value

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from data_layer import backend; print(backend.load_prices.__module__)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return result.stdout.strip()


def test_default_backend_is_yfinance():
    assert _backend_module(None) == "data_layer.yfinance_provider"


def test_local_backend_selection():
    assert _backend_module("local") == "data_layer.local"


def test_sharadar_backend_selection():
    assert _backend_module("sharadar") == "data_layer.sharadar"
