"""W80 / P0 #12 — R-201 live local-model benchmark tests."""

from __future__ import annotations

import pytest

try:
    import torch  # type: ignore  # noqa: F401
    import transformers  # type: ignore  # noqa: F401
    _HAS_HF = True
except Exception:  # noqa: BLE001
    _HAS_HF = False


def test_w80_r201_schema_versioned():
    from coordpy.r201_benchmark import R201_SCHEMA_VERSION
    assert R201_SCHEMA_VERSION == "coordpy.r201_benchmark.v1"


def test_w80_r201_returns_deterministic_unavailable_report_without_hf():
    """When transformers is missing, R-201 must produce a
    deterministic 'unavailable' report rather than silently
    passing or crashing."""
    if _HAS_HF:
        pytest.skip(
            "transformers available — this exercises the "
            "missing-deps path")
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    assert r["transformers_available"] is False
    assert bool(r["all_pass"]) is False
    assert r["cells"]["H1400"] is False


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_all_pass_against_live_model():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    assert r["transformers_available"] is True
    assert bool(r["all_pass"]), (
        f"r201 failures: "
        f"{dict((k, v) for k, v in r['cells'].items() if not v)}")
    # Total number of H-bars in R-201.
    assert len(r["cells"]) == 22
    # All H-bars in [1400, 1421] must be present.
    for h in range(1400, 1422):
        assert f"H{h}" in r["cells"]


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_replay_is_byte_identical_within_fp32_tol():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    # The replay-vs-recompute summary should report byte-identity.
    summary = r["metrics"]["replay_vs_recompute_summary"]
    assert bool(summary["replay_byte_identical"])
    assert float(summary["max_abs_diff_last_logits"]) < 5e-3


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_substrate_routing_saves_token_work():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    metrics = r["metrics"]
    # Substrate routing must save strictly less token work than
    # transcript-only on a long-prompt scenario.
    assert int(metrics["token_work_substrate_routed"]) < int(
        metrics["token_work_transcript_only"])
    # And the saving ratio must be at least 30 % on the default
    # prompt.
    assert float(
        metrics["substrate_routed_token_saving_ratio"]) >= 0.3


@pytest.mark.skipif(
    not _HAS_HF,
    reason="transformers / torch not installed")
def test_w80_r201_conformance_passes_on_live_runtime():
    from coordpy.r201_benchmark import run_r201
    r = run_r201()
    assert int(r["metrics"]["conformance_fail_count"]) == 0
    assert int(r["metrics"]["conformance_pass_count"]) == 12
