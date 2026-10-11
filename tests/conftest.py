# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest hooks for the test suite."""

from __future__ import annotations

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--heavy-data-root",
        default=None,
        metavar="PATH",
        help=(
            "Persistent directory for heavy-test data (reference genome, task datasets). "
            "Reused across runs so data is only downloaded once. "
            "Defaults to a fresh temp directory when not provided."
        ),
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Skip baseline-update tests unless that test is invoked explicitly by name."""
    explicit_update = any(
        "test_heavy_update_baseline" in arg for arg in config.args
    )
    if explicit_update:
        return

    skip = pytest.mark.skip(
        reason=(
            "Baseline update runs only when invoked by name: "
            "pytest tests/e2e/test_heavy.py::test_heavy_update_baseline"
        )
    )
    for item in items:
        if item.name == "test_heavy_update_baseline":
            item.add_marker(skip)
