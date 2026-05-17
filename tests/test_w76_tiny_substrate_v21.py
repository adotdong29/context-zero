"""W76 tests — tiny_substrate_v21."""

from __future__ import annotations

from coordpy.tiny_substrate_v16 import record_restart_event_v16
from coordpy.tiny_substrate_v20 import (
    record_compound_chain_window_v20,
)
from coordpy.tiny_substrate_v21 import (
    W76_DEFAULT_V21_N_LAYERS, W76_REPAIR_CHAIN_THEN_RESTART,
    W76_REPAIR_LABELS_V21,
    build_default_tiny_substrate_v21,
    emit_tiny_substrate_v21_forward_witness,
    forward_tiny_substrate_v21,
    record_post_compound_chain_restart_window_v21,
    substrate_chain_then_restart_pressure_throttle_v21,
    substrate_chain_then_restart_repair_dominance_flops_v21,
    tokenize_bytes_v21,
)


def test_v21_substrate_has_23_layers() -> None:
    p = build_default_tiny_substrate_v21()
    ids = tokenize_bytes_v21("w76", max_len=4)
    trace, _ = forward_tiny_substrate_v21(p, ids)
    assert trace.v21_gate_score_per_layer.shape[0] == 23
    assert W76_DEFAULT_V21_N_LAYERS == 23


def test_v21_repair_labels_include_chain_then_restart() -> None:
    assert len(W76_REPAIR_LABELS_V21) == 13
    assert W76_REPAIR_LABELS_V21[
        W76_REPAIR_CHAIN_THEN_RESTART] == (
            "restart_after_compound_chain_repair")


def test_v21_chain_then_restart_cid_content_addressed() -> None:
    p = build_default_tiny_substrate_v21()
    ids = tokenize_bytes_v21("w76-ctr-cid", max_len=12)
    _, cache_a = forward_tiny_substrate_v21(p, ids)
    _, cache_b = forward_tiny_substrate_v21(p, ids)
    # Same setup, same CID.
    assert (
        str(cache_a.compound_chain_then_restart_trajectory_cid)
        == str(
            cache_b.compound_chain_then_restart_trajectory_cid))
    # Record events and re-run — CID changes.
    record_post_compound_chain_restart_window_v21(
        cache_a, compound_chain_repair_turn=10,
        restart_turn=15,
        post_compound_chain_restart_window_turns=10,
        role="r", branch_id="b")
    _, cache_a2 = forward_tiny_substrate_v21(
        p, ids, v21_kv_cache=cache_a)
    assert (
        str(cache_a2.compound_chain_then_restart_trajectory_cid)
        != str(
            cache_b.compound_chain_then_restart_trajectory_cid))


def test_v21_chain_then_restart_label_fires() -> None:
    p = build_default_tiny_substrate_v21()
    ids = tokenize_bytes_v21("w76-ctr-label", max_len=12)
    _, cache = forward_tiny_substrate_v21(p, ids)
    v20 = cache.v20_cache
    v19 = v20.v19_cache
    v18 = v19.v18_cache
    # Compound-chain window is required.
    record_compound_chain_window_v20(
        v20, replacement_turn=3, delayed_repair_turn=4,
        rejoin_turn=10, compound_chain_window_turns=8,
        role="p", branch_id="b")
    # Restart event required.
    record_restart_event_v16(
        v18.v17_cache.v16_cache, turn=15,
        restart_kind="post_compound_chain_restart", role="p")
    # Post-compound-chain-restart window.
    record_post_compound_chain_restart_window_v21(
        cache, compound_chain_repair_turn=10, restart_turn=15,
        post_compound_chain_restart_window_turns=8,
        role="p", branch_id="b")
    trace, cache = forward_tiny_substrate_v21(
        p, ids, v21_kv_cache=cache,
        compound_chain_then_restart_pressure=0.85)
    w = emit_tiny_substrate_v21_forward_witness(trace, cache)
    assert int(w.compound_chain_then_restart_repair_l1) >= 1


def test_v21_chain_then_restart_dominance_flops_saves() -> None:
    r = substrate_chain_then_restart_repair_dominance_flops_v21(
        n_tokens=128, n_repairs=12)
    assert r["saving_ratio"] >= 0.9
    assert r["recompute_flops"] > r[
        "chain_then_restart_dominance_flops"]


def test_v21_chain_then_restart_pressure_throttle_saves() -> None:
    r = substrate_chain_then_restart_pressure_throttle_v21(
        visible_token_budget=32, baseline_token_cost=512,
        post_compound_chain_restart_window_turns=4)
    assert r["chain_then_restart_pressure_active"]
    assert r["saving_tokens"] > 0
