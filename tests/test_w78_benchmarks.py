"""W78 — R-193..R-196 benchmark families test."""

from __future__ import annotations


def test_r193_plane_a_v11():
    from coordpy.r193_benchmark import run_r193
    r = run_r193(seeds=[1, 2, 3])
    assert bool(r["all_pass"])
    assert int(len(r["cells"])) == 10


def test_r194_plane_b_v23():
    from coordpy.r194_benchmark import run_r194
    r = run_r194(seeds=[1, 2, 3])
    assert bool(r["all_pass"])
    assert int(len(r["cells"])) == 18


def test_r195_multi_agent_18_regime():
    from coordpy.r195_benchmark import run_r195
    r = run_r195(seeds=[1, 2, 3])
    assert bool(r["all_pass"])
    assert int(r["n_regimes"]) == 18


def test_r196_handoff_v10_and_falsifiers():
    from coordpy.r196_benchmark import run_r196
    r = run_r196(seeds=[1, 2, 3])
    assert bool(r["all_pass"])
    assert int(len(r["cells"])) == 16


def test_r193_two_consecutive_runs():
    """Stability: two consecutive runs both pass."""
    from coordpy.r193_benchmark import run_r193
    r1 = run_r193(seeds=[1, 2, 3])
    r2 = run_r193(seeds=[1, 2, 3])
    assert bool(r1["all_pass"]) and bool(r2["all_pass"])


def test_r194_two_consecutive_runs():
    from coordpy.r194_benchmark import run_r194
    r1 = run_r194(seeds=[1, 2, 3])
    r2 = run_r194(seeds=[1, 2, 3])
    assert bool(r1["all_pass"]) and bool(r2["all_pass"])
