"""W77 tests — R189..R192 benchmark families end-to-end."""

from __future__ import annotations

from coordpy.r189_benchmark import R189_SCHEMA_VERSION, run_r189
from coordpy.r190_benchmark import R190_SCHEMA_VERSION, run_r190
from coordpy.r191_benchmark import R191_SCHEMA_VERSION, run_r191
from coordpy.r192_benchmark import R192_SCHEMA_VERSION, run_r192


def test_r189_all_pass() -> None:
    res = run_r189(seeds=[1, 2, 3, 4, 5])
    assert res["schema"] == R189_SCHEMA_VERSION
    assert res["all_pass"], f"R189 cells: {res['cells']}"


def test_r190_all_pass() -> None:
    res = run_r190(seeds=[1, 2, 3, 4, 5])
    assert res["schema"] == R190_SCHEMA_VERSION
    assert res["all_pass"], f"R190 cells: {res['cells']}"


def test_r191_all_pass() -> None:
    res = run_r191(seeds=[1, 2, 3, 4, 5])
    assert res["schema"] == R191_SCHEMA_VERSION
    assert res["n_regimes"] == 17
    assert res["all_pass"], f"R191 cells: {res['cells']}"


def test_r192_all_pass() -> None:
    res = run_r192(seeds=[1, 2, 3, 4, 5])
    assert res["schema"] == R192_SCHEMA_VERSION
    assert res["all_pass"], f"R192 cells: {res['cells']}"
