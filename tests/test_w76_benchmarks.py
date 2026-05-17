"""W76 benchmark family tests — R-185..R-188 all-pass."""

from __future__ import annotations

import pytest

from coordpy.r185_benchmark import run_r185
from coordpy.r186_benchmark import run_r186
from coordpy.r187_benchmark import run_r187
from coordpy.r188_benchmark import run_r188


@pytest.mark.parametrize("seed_set", [
    [1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14],
    [100, 101, 102, 103, 104],
])
def test_r185_all_pass(seed_set: list[int]) -> None:
    r = run_r185(seeds=seed_set)
    assert r["all_pass"], r["cells"]


@pytest.mark.parametrize("seed_set", [
    [1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14],
    [100, 101, 102, 103, 104],
])
def test_r186_all_pass(seed_set: list[int]) -> None:
    r = run_r186(seeds=seed_set)
    assert r["all_pass"], r["cells"]


@pytest.mark.parametrize("seed_set", [
    [1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14],
    [100, 101, 102, 103, 104],
])
def test_r187_all_pass(seed_set: list[int]) -> None:
    r = run_r187(seeds=seed_set)
    assert r["all_pass"], r["cells"]
    assert r["n_regimes"] == 16


@pytest.mark.parametrize("seed_set", [
    [1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14],
    [100, 101, 102, 103, 104],
])
def test_r188_all_pass(seed_set: list[int]) -> None:
    r = run_r188(seeds=seed_set)
    assert r["all_pass"], r["cells"]
