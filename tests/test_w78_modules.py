"""W78 — focused module tests."""

from __future__ import annotations

import pytest


def test_w78_substrate_v23_smoke():
    from coordpy.tiny_substrate_v23 import (
        W78_REPAIR_LABELS_V23,
        build_default_tiny_substrate_v23,
        emit_tiny_substrate_v23_forward_witness,
        forward_tiny_substrate_v23,
        record_long_horizon_reconstruction_window_v23,
        tokenize_bytes_v23,
    )
    p = build_default_tiny_substrate_v23()
    ids = tokenize_bytes_v23("w78 smoke", max_len=8)
    trace, cache = forward_tiny_substrate_v23(p, ids)
    assert len(W78_REPAIR_LABELS_V23) == 15
    record_long_horizon_reconstruction_window_v23(
        cache, compound_chain_failure_turn=30,
        reconstruction_request_turn=88,
        long_horizon_blackout_window_turns=58)
    trace, cache = forward_tiny_substrate_v23(
        p, ids, v23_kv_cache=cache,
        long_horizon_reconstruction_pressure=0.9)
    w = emit_tiny_substrate_v23_forward_witness(trace, cache)
    assert w.long_horizon_reconstruction_l1 > 0
    assert isinstance(w.cid(), str) and len(w.cid()) == 64


def test_w78_kv_bridge_v23():
    from coordpy.kv_bridge_v23 import (
        KVBridgeV23Projection,
        compute_long_horizon_reconstruction_fingerprint_v23,
        emit_kv_bridge_v23_witness,
        probe_kv_bridge_v23_long_horizon_reconstruction_falsifier,
    )
    from coordpy.w77_team import W77Params
    p = W77Params.build_default(seed=78100)
    kv = KVBridgeV23Projection.init_from_v22(
        p.kv_bridge_v22, seed_v23=78101)
    fp = compute_long_horizon_reconstruction_fingerprint_v23(
        role="r",
        post_restart_replacement_trajectory_cid="prc",
        long_horizon_reconstruction_trajectory_cid="lhr",
        long_horizon_reconstruction_count=2,
        long_horizon_blackout_window_turns=50)
    assert len(fp) == 156
    fals = (
        probe_kv_bridge_v23_long_horizon_reconstruction_falsifier(
            long_horizon_reconstruction_pressure_flag=1))
    assert float(fals.falsifier_score) == 0.0  # honest
    w = emit_kv_bridge_v23_witness(
        projection=kv,
        long_horizon_reconstruction_falsifier=fals,
        long_horizon_reconstruction_fingerprint=fp)
    assert isinstance(w.cid(), str)


def test_w78_cache_v21():
    import numpy as np

    from coordpy.cache_controller_v21 import (
        CacheControllerV21, fit_eighteen_objective_ridge_v21,
    )
    cc = CacheControllerV21.init(fit_seed=78110)
    rng = np.random.default_rng(78111)
    X = rng.standard_normal((20, 4)).tolist()
    ys = [
        rng.standard_normal(20).tolist() for _ in range(18)]
    cc, rep = fit_eighteen_objective_ridge_v21(
        controller=cc, train_features=X,
        target_drop_oracle=ys[0],
        target_retrieval_relevance=ys[1],
        target_hidden_wins=ys[2],
        target_replay_dominance=ys[3],
        target_team_task_success=ys[4],
        target_team_failure_recovery=ys[5],
        target_branch_merge=ys[6],
        target_partial_contradiction=ys[7],
        target_multi_branch_rejoin=ys[8],
        target_budget_primary=ys[9],
        target_restart_dominance=ys[10],
        target_delayed_rejoin_after_restart=ys[11],
        target_replacement_after_ctr=ys[12],
        target_compound_repair=ys[13],
        target_compound_chain_repair=ys[14],
        target_chain_then_restart_repair=ys[15],
        target_post_restart_replacement_repair=ys[16],
        target_long_horizon_reconstruction_repair=ys[17])
    assert int(rep.n_objectives) == 18
    assert cc.eighteen_objective_head is not None


def test_w78_replay_v19():
    from coordpy.replay_controller_v19 import (
        ReplayControllerV19, W78_REPLAY_REGIMES_V19,
        W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS,
    )
    assert len(W78_REPLAY_REGIMES_V19) == 26
    assert len(
        W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS) == 16
    rc = ReplayControllerV19.init()
    assert isinstance(rc.cid(), str)


def test_w78_persistent_v30_lhr_v30():
    from coordpy.long_horizon_retention_v30 import (
        LongHorizonReconstructionV30Head,
        W78_DEFAULT_LHR_V30_MAX_K,
    )
    from coordpy.persistent_latent_v30 import (
        W78_DEFAULT_V30_MAX_CHAIN_WALK_DEPTH,
        W78_DEFAULT_V30_N_LAYERS,
        PersistentLatentStateV30Chain,
    )
    assert int(W78_DEFAULT_V30_N_LAYERS) == 29
    assert int(W78_DEFAULT_V30_MAX_CHAIN_WALK_DEPTH) == 33554432
    head = LongHorizonReconstructionV30Head.init()
    assert int(head.max_k) == int(W78_DEFAULT_LHR_V30_MAX_K) == 2048
    chain = PersistentLatentStateV30Chain.empty()
    assert isinstance(chain.cid(), str)


def test_w78_consensus_v24():
    from coordpy.consensus_fallback_controller_v24 import (
        W78_CONSENSUS_V24_STAGES,
        ConsensusFallbackControllerV24,
    )
    assert len(W78_CONSENSUS_V24_STAGES) >= 42
    c = ConsensusFallbackControllerV24.init()
    assert isinstance(c.cid(), str)


def test_w78_substrate_adapter_v23():
    from coordpy.substrate_adapter_v23 import (
        W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL,
        probe_all_v23_adapters,
        probe_tiny_substrate_v23_adapter,
    )
    cap = probe_tiny_substrate_v23_adapter()
    assert cap.tier == W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL
    matrix = probe_all_v23_adapters()
    assert matrix.has_v23_full()


def test_w78_masc_v14_baseline():
    from coordpy.multi_agent_substrate_coordinator_v14 import (
        MultiAgentSubstrateCoordinatorV14,
        W78_MASC_V14_REGIMES,
    )
    assert len(W78_MASC_V14_REGIMES) == 18
    m = MultiAgentSubstrateCoordinatorV14()
    _, agg = m.run_batch(seeds=[1, 2, 3], regime="baseline")
    assert float(agg.v23_beats_v22_rate) >= 0.5
    assert float(agg.tsc_v23_beats_tsc_v22_rate) >= 0.5


def test_w78_tcc_v13():
    from coordpy.team_consensus_controller_v13 import (
        TeamConsensusControllerV13,
        emit_team_consensus_controller_v13_witness,
    )
    tcc = TeamConsensusControllerV13()
    d = tcc.decide_v13(
        regime="baseline",
        agent_guesses=[0.5, 0.4, 0.3, 0.2],
        agent_confidences=[0.8, 0.7, 0.7, 0.6],
        substrate_replay_trust=0.7,
        long_horizon_reconstruction_pressure=0.9,
        agent_long_horizon_reconstruction_recovery_flags=[
            1, 0, 1, 0])
    assert "decision" in d
    w = emit_team_consensus_controller_v13_witness(tcc)
    assert isinstance(w.cid(), str)


def test_w78_deep_v23():
    from coordpy.deep_substrate_hybrid_v22 import (
        DeepSubstrateHybridV22ForwardWitness,
    )
    from coordpy.deep_substrate_hybrid_v23 import (
        DeepSubstrateHybridV23,
        deep_substrate_hybrid_v23_forward,
    )
    h = DeepSubstrateHybridV23()
    v22w = DeepSubstrateHybridV22ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v22.v1",
        hybrid_cid="", inner_v21_witness_cid="",
        twenty_two_way=True,
        cache_controller_v20_fired=True,
        replay_controller_v18_fired=True,
        post_restart_replacement_trajectory_active=True,
        post_restart_replacement_repair_active=True,
        team_consensus_controller_v12_active=True,
        post_restart_replacement_trajectory_cid="prc",
        post_restart_replacement_repair_l1=1,
        post_restart_replacement_pressure_gate_mean=0.5)
    # Without cache/replay it should NOT trigger 23-way.
    w = deep_substrate_hybrid_v23_forward(
        hybrid=h, v22_witness=v22w,
        long_horizon_reconstruction_trajectory_cid="lhr",
        long_horizon_reconstruction_repair_l1=1,
        n_team_consensus_v13_invocations=1)
    assert not bool(w.twenty_three_way)


def test_w78_hosted_router_v11():
    from coordpy.hosted_router_controller import (
        default_hosted_registry,
    )
    from coordpy.hosted_router_controller_v11 import (
        HostedRouterControllerV11,
        emit_hosted_router_controller_v11_witness,
    )
    reg = default_hosted_registry()
    r = HostedRouterControllerV11.init(reg, {})
    w = emit_hosted_router_controller_v11_witness(r)
    assert isinstance(w.cid(), str)
