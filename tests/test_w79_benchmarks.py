"""W79 — R-197..R-200 benchmark runs."""

from __future__ import annotations

import pytest


def test_r197_all_pass():
    from coordpy.r197_benchmark import run_r197
    r = run_r197(seeds=[1, 2, 3])
    assert bool(r["all_pass"]), (
        f"r197 failures: {dict((k, v) for k, v in r['cells'].items() if not v)}")


def test_r198_all_pass():
    from coordpy.r198_benchmark import run_r198
    r = run_r198(seeds=[1, 2, 3])
    assert bool(r["all_pass"]), (
        f"r198 failures: {dict((k, v) for k, v in r['cells'].items() if not v)}")


def test_r199_all_pass():
    from coordpy.r199_benchmark import run_r199
    r = run_r199(seeds=[1, 2, 3])
    assert bool(r["all_pass"]), (
        f"r199 failures: {dict((k, v) for k, v in r['cells'].items() if not v)}")
    assert int(r["n_regimes"]) == 19


def test_r200_all_pass():
    from coordpy.r200_benchmark import run_r200
    r = run_r200(seeds=[1, 2, 3])
    assert bool(r["all_pass"]), (
        f"r200 failures: {dict((k, v) for k, v in r['cells'].items() if not v)}")
