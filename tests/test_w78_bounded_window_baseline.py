"""W78 — bounded-window baseline (anti-goal falsifier) tests.

These tests prove the load-bearing W78 anti-goal: fixed-k
transcript baselines CANNOT answer reconstruction queries whose
source events lie outside their visible window. The W78 substrate
must beat them on the same queries.
"""

from __future__ import annotations


def test_bounded_window_baseline_fails_outside_window():
    from coordpy.bounded_window_baseline_v1 import (
        BoundedWindowQuery,
        build_default_bounded_window_baselines,
        run_bounded_window_falsifier,
    )
    baselines = build_default_bounded_window_baselines()
    # Query whose source is 100 turns ago — outside every fixed-k.
    q = BoundedWindowQuery(
        query_id="q1", current_turn=200, source_turn=100,
        expected_event_cid="event_x")
    _, fals = run_bounded_window_falsifier(
        baselines=baselines, query=q)
    assert bool(fals.all_fixed_k_failed)
    assert int(fals.fixed_k_failure_count) >= 4


def test_bounded_window_baseline_succeeds_inside_window():
    from coordpy.bounded_window_baseline_v1 import (
        BoundedWindowQuery,
        build_default_bounded_window_baselines,
        run_bounded_window_falsifier,
    )
    baselines = build_default_bounded_window_baselines()
    # Query whose source is 2 turns ago — inside every fixed-k.
    q = BoundedWindowQuery(
        query_id="q2", current_turn=100, source_turn=98,
        expected_event_cid="event_y")
    outs, fals = run_bounded_window_falsifier(
        baselines=baselines, query=q)
    # Every fixed-k succeeds (since 100-98=2 < k for k=4,8,16,32).
    assert not bool(fals.all_fixed_k_failed)
    for o in outs:
        if not o.rolling_summary:
            assert bool(o.success)


def test_bounded_window_insufficiency_proof():
    from coordpy.bounded_window_baseline_v1 import (
        build_default_bounded_window_baselines,
        prove_bounded_window_insufficient,
    )
    baselines = build_default_bounded_window_baselines()
    # 100 turns horizon ≥ max_k = 32 → proven.
    p = prove_bounded_window_insufficient(
        query_horizon_turns=100, baselines=baselines)
    assert bool(p.proven)
    assert int(p.max_baseline_k) == 32
    # 10 turns horizon < max_k → not proven (some baseline can
    # still succeed).
    p2 = prove_bounded_window_insufficient(
        query_horizon_turns=10, baselines=baselines)
    assert not bool(p2.proven)


def test_w78_substrate_beats_bounded_window_baseline():
    """W78 substrate succeeds on the same query bounded-window
    fails on."""
    from coordpy.bounded_window_baseline_v1 import (
        BoundedWindowQuery,
        build_default_bounded_window_baselines,
        run_bounded_window_falsifier,
    )
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
        build_default_long_horizon_reconstruction_carrier,
        reconstruct_long_horizon_event,
    )
    baselines = build_default_bounded_window_baselines()
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=256, seed=78200)
    # Pick a source turn that's outside every fixed-k.
    known = carrier.entries[50]
    bw_q = BoundedWindowQuery(
        query_id="bw1", current_turn=250,
        source_turn=int(known.turn_index),
        expected_event_cid=str(known.event_cid))
    _, fals = run_bounded_window_falsifier(
        baselines=baselines, query=bw_q)
    assert bool(fals.all_fixed_k_failed)
    # W78 substrate reconstructs the event.
    lhr_q = LongHorizonReconstructionQuery(
        query_id="lhr1", source_turn=int(known.turn_index),
        current_turn=250)
    out = reconstruct_long_horizon_event(
        carrier=carrier, query=lhr_q, visible_tokens_used=4)
    assert bool(out.success)
    assert str(out.reconstructed_event_cid) == str(
        known.event_cid)


def test_bounded_window_baseline_witness():
    from coordpy.bounded_window_baseline_v1 import (
        BoundedWindowQuery,
        build_default_bounded_window_baselines,
        emit_bounded_window_baseline_witness,
        prove_bounded_window_insufficient,
        run_bounded_window_falsifier,
    )
    baselines = build_default_bounded_window_baselines()
    q = BoundedWindowQuery(
        query_id="q", current_turn=200, source_turn=20,
        expected_event_cid="ev")
    _, fals = run_bounded_window_falsifier(
        baselines=baselines, query=q)
    proof = prove_bounded_window_insufficient(
        query_horizon_turns=180, baselines=baselines)
    w = emit_bounded_window_baseline_witness(
        baselines=baselines, proof=proof, falsifier=fals)
    assert int(w.n_fixed_k) == 4
    assert tuple(w.fixed_k_values) == (4, 8, 16, 32)
    assert isinstance(w.cid(), str)
