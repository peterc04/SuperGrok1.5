"""Pytest configuration for the test suite.

Registers the ``hw`` marker used to gate hardware (GPU/multi-GPU) tests so
that ``pytest -m "not hw"`` (the CPU CI path) runs cleanly and the marker does
not raise ``PytestUnknownMarkWarning``. The hardware half of the suite is run
from the runbook (HARDWARE_VALIDATION.md) on real accelerators.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hw: hardware-gated test (GPU / multi-GPU); skipped on CPU CI, run "
        "from HARDWARE_VALIDATION.md on real accelerators.",
    )
