"""W78 — long-horizon reconstruction substrate tests."""

from __future__ import annotations


def test_reconstruction_succeeds_for_in_carrier_turn():
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
        build_default_long_horizon_reconstruction_carrier,
        reconstruct_long_horizon_event,
    )
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=256, seed=78300)
    known = carrier.entries[42]
    q = LongHorizonReconstructionQuery(
        query_id="q", source_turn=42, current_turn=240)
    out = reconstruct_long_horizon_event(
        carrier=carrier, query=q)
    assert bool(out.success)
    assert str(out.reconstructed_event_cid) == str(
        known.event_cid)
    assert int(out.horizon_turns) == 198


def test_reconstruction_deterministic_under_repeat():
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
        build_default_long_horizon_reconstruction_carrier,
        reconstruct_long_horizon_event,
    )
    carrier1 = build_default_long_horizon_reconstruction_carrier(
        n_events=128, seed=78301)
    carrier2 = build_default_long_horizon_reconstruction_carrier(
        n_events=128, seed=78301)
    assert str(carrier1.cid()) == str(carrier2.cid())
    q = LongHorizonReconstructionQuery(
        query_id="q", source_turn=10, current_turn=120)
    o1 = reconstruct_long_horizon_event(
        carrier=carrier1, query=q)
    o2 = reconstruct_long_horizon_event(
        carrier=carrier2, query=q)
    assert str(o1.reconstructed_event_cid) == str(
        o2.reconstructed_event_cid)


def test_reconstruction_fails_out_of_carrier():
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
        build_default_long_horizon_reconstruction_carrier,
        reconstruct_long_horizon_event,
    )
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=32, seed=78302)
    q = LongHorizonReconstructionQuery(
        query_id="q", source_turn=1000, current_turn=1100)
    out = reconstruct_long_horizon_event(
        carrier=carrier, query=q)
    assert not bool(out.success)


def test_economics_substrate_beats_recompute():
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
        build_default_long_horizon_reconstruction_carrier,
        report_reconstruction_vs_recompute_economics,
    )
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=256, seed=78303)
    q = LongHorizonReconstructionQuery(
        query_id="q", source_turn=10, current_turn=250)
    rep = report_reconstruction_vs_recompute_economics(
        query=q, carrier=carrier)
    # O(log n) Merkle walks << O(horizon) per-token replay.
    assert float(rep.saving_ratio) >= 0.90


def test_reconstruction_witness():
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
        build_default_long_horizon_reconstruction_carrier,
        emit_long_horizon_reconstruction_witness,
        reconstruct_long_horizon_event,
        report_reconstruction_vs_recompute_economics,
    )
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=64, seed=78304)
    queries = [
        LongHorizonReconstructionQuery(
            query_id=f"q{i}", source_turn=i * 6,
            current_turn=120)
        for i in range(5)]
    outs = [
        reconstruct_long_horizon_event(
            carrier=carrier, query=q) for q in queries]
    econ = report_reconstruction_vs_recompute_economics(
        query=queries[-1], carrier=carrier)
    w = emit_long_horizon_reconstruction_witness(
        carrier=carrier, outcomes=outs, economics=econ)
    assert int(w.n_queries_succeeded) == 5
    assert float(w.success_rate) == 1.0
    assert isinstance(w.cid(), str)
