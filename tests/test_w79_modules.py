"""W79 — focused module tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_w79_substrate_v24_smoke():
    from coordpy.tiny_substrate_v24 import (
        W79_REPAIR_LABELS_V24,
        build_default_tiny_substrate_v24,
        emit_tiny_substrate_v24_forward_witness,
        forward_tiny_substrate_v24,
        record_replacement_then_restart_after_long_delay_window_v24,
        tokenize_bytes_v24,
    )
    p = build_default_tiny_substrate_v24()
    ids = tokenize_bytes_v24("w79 smoke", max_len=8)
    trace, cache = forward_tiny_substrate_v24(
        p, ids,
        replacement_then_restart_after_long_delay_pressure=0.95)
    assert len(W79_REPAIR_LABELS_V24) == 16
    record_replacement_then_restart_after_long_delay_window_v24(
        cache, long_delay_blackout_start_turn=30,
        replacement_turn=45, restart_turn=70,
        reconstruction_request_turn=95)
    trace, cache = forward_tiny_substrate_v24(
        p, ids, v24_kv_cache=cache,
        replacement_then_restart_after_long_delay_pressure=0.95)
    w = emit_tiny_substrate_v24_forward_witness(p, trace, cache)
    assert w.replacement_then_restart_after_long_delay_l1 > 0
    assert isinstance(w.cid(), str) and len(w.cid()) == 64


def test_w79_kv_bridge_v24():
    from coordpy.kv_bridge_v24 import (
        KVBridgeV24Projection,
        W79_KV_BRIDGE_V24_FINGERPRINT_DIM,
        compute_replacement_then_restart_after_long_delay_fingerprint_v24,
        emit_kv_bridge_v24_witness,
        probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier,
    )
    from coordpy.w78_team import W78Params
    p = W78Params.build_default(seed=78100)
    kv = KVBridgeV24Projection.init_from_v23(
        p.kv_bridge_v23, seed_v24=78101)
    fp = (
        compute_replacement_then_restart_after_long_delay_fingerprint_v24(
            role="r",
            long_horizon_reconstruction_trajectory_cid="lhr",
            replacement_then_restart_after_long_delay_trajectory_cid="rtrld",
            replacement_then_restart_after_long_delay_count=2,
            long_delay_blackout_window_turns=50))
    assert len(fp) == int(W79_KV_BRIDGE_V24_FINGERPRINT_DIM)
    fals = (
        probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier(
            replacement_then_restart_after_long_delay_pressure_flag=1))
    assert float(fals.falsifier_score) == 0.0  # honest
    w = emit_kv_bridge_v24_witness(
        projection=kv,
        replacement_then_restart_after_long_delay_falsifier=fals,
        replacement_then_restart_after_long_delay_fingerprint=fp)
    assert isinstance(w.cid(), str)


def test_w79_cache_v22():
    from coordpy.cache_controller_v22 import (
        CacheControllerV22,
        W79_CACHE_CONTROLLER_V22_N_OBJECTIVES,
        fit_nineteen_objective_ridge_v22,
    )
    cc = CacheControllerV22.init(fit_seed=79110)
    rng = np.random.default_rng(79111)
    X = rng.standard_normal((20, 4)).tolist()
    targets = [
        rng.standard_normal(20).tolist() for _ in range(
            int(W79_CACHE_CONTROLLER_V22_N_OBJECTIVES))]
    cc, rep = fit_nineteen_objective_ridge_v22(
        controller=cc, train_features=X,
        targets_per_objective=targets)
    assert int(rep.n_objectives) == 19
    assert cc.nineteen_objective_head is not None


def test_w79_replay_v20():
    from coordpy.replay_controller import ReplayCandidate
    from coordpy.replay_controller_v20 import (
        ReplayControllerV20,
        W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS,
        W79_REPLAY_REGIMES_V20,
        emit_replay_controller_v20_witness,
        fit_replay_controller_v20_per_role,
        fit_replay_v20_replacement_then_restart_after_long_delay_aware_routing_head,
    )
    assert len(W79_REPLAY_REGIMES_V20) == 27
    assert (
        len(W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS)
        == 17)
    rc = ReplayControllerV20.init()
    cands = {
        r: [ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0)]
        for r in W79_REPLAY_REGIMES_V20}
    decs = {r: ["choose_reuse"] for r in W79_REPLAY_REGIMES_V20}
    rc, audit = fit_replay_controller_v20_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    assert int(audit["n_regimes_fit"]) == 27
    rng = np.random.default_rng(79111)
    X = rng.standard_normal((64, 10)).tolist()
    labels = list(
        W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS
        * (64 // 17 + 1))[:64]
    rc, audit2 = (
        fit_replay_v20_replacement_then_restart_after_long_delay_aware_routing_head(
            controller=rc,
            train_team_features=X,
            train_routing_labels=labels))
    assert int(audit2["n_labels"]) == 17
    w = emit_replay_controller_v20_witness(rc)
    assert w.routing_head_present
    assert isinstance(w.cid(), str)


def test_w79_consensus_v25():
    from coordpy.consensus_fallback_controller_v25 import (
        ConsensusFallbackControllerV25,
        W79_CONSENSUS_V25_N_STAGES,
        emit_consensus_v25_witness,
    )
    c = ConsensusFallbackControllerV25.init(
        k_required=2, cosine_floor=0.6,
        trust_threshold=0.5,
        multi_branch_rejoin_threshold=0.5,
        silent_corruption_threshold=0.5,
        repair_dominance_threshold=0.5,
        budget_primary_threshold=0.5,
        restart_aware_threshold=0.5,
        delayed_repair_threshold=0.5,
        rejoin_pressure_threshold=0.5,
        delayed_rejoin_threshold=0.5,
        replacement_pressure_threshold=0.5,
        replacement_after_ctr_threshold=0.5,
        compound_repair_threshold=0.5,
        compound_repair_drtr_threshold=0.5,
        compound_chain_repair_threshold=0.5,
        compound_repair_rtr_threshold=0.5,
        chain_then_restart_threshold=0.5,
        post_compound_chain_restart_threshold=0.5,
        post_restart_replacement_threshold=0.5,
        post_restart_replacement_best_parent_threshold=0.5,
        long_horizon_reconstruction_threshold=0.5,
        long_horizon_reconstruction_best_parent_threshold=0.5,
        replacement_then_restart_after_long_delay_threshold=0.5,
        replacement_then_restart_after_long_delay_best_parent_threshold=0.5)
    assert int(W79_CONSENSUS_V25_N_STAGES) == 44
    w = emit_consensus_v25_witness(c)
    assert isinstance(w.cid(), str)


def test_w79_persistent_v31():
    from coordpy.persistent_latent_v31 import (
        W79_DEFAULT_V31_MAX_CHAIN_WALK_DEPTH,
        W79_DEFAULT_V31_N_LAYERS,
        PersistentLatentStateV31Chain,
        emit_persistent_v31_witness,
    )
    assert int(W79_DEFAULT_V31_N_LAYERS) == 30
    assert int(W79_DEFAULT_V31_MAX_CHAIN_WALK_DEPTH) == 67108864
    chain = PersistentLatentStateV31Chain.empty()
    w = emit_persistent_v31_witness(chain)
    assert int(w.n_layers) == 30


def test_w79_lhr_v31():
    from coordpy.long_horizon_retention_v31 import (
        W79_DEFAULT_LHR_V31_MAX_K,
        LongHorizonReconstructionV31Head,
        emit_lhr_v31_witness,
    )
    head = LongHorizonReconstructionV31Head.init(seed=200)
    assert int(W79_DEFAULT_LHR_V31_MAX_K) == 4096
    w = emit_lhr_v31_witness(
        head, carrier=[0.1] * 6, k=16,
        long_horizon_reconstruction_indicator=[0.95] * 8,
        replacement_then_restart_after_long_delay_indicator=[
            0.97] * 8)
    assert int(w.n_heads) == 30
    assert bool(w.thirty_way_runs)


def test_w79_mlsc_v27():
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3,
    )
    from coordpy.mergeable_latent_capsule_v26 import (
        wrap_v25_as_v26,
    )
    from coordpy.mergeable_latent_capsule_v27 import (
        emit_mlsc_v27_witness, wrap_v26_as_v27,
    )
    # Build a minimal V27 wrapper.
    from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
    from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
    from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
    from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
    from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
    from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
    from coordpy.mergeable_latent_capsule_v10 import wrap_v9_as_v10
    from coordpy.mergeable_latent_capsule_v11 import wrap_v10_as_v11
    from coordpy.mergeable_latent_capsule_v12 import wrap_v11_as_v12
    from coordpy.mergeable_latent_capsule_v13 import wrap_v12_as_v13
    from coordpy.mergeable_latent_capsule_v14 import wrap_v13_as_v14
    from coordpy.mergeable_latent_capsule_v15 import wrap_v14_as_v15
    from coordpy.mergeable_latent_capsule_v16 import wrap_v15_as_v16
    from coordpy.mergeable_latent_capsule_v17 import wrap_v16_as_v17
    from coordpy.mergeable_latent_capsule_v18 import wrap_v17_as_v18
    from coordpy.mergeable_latent_capsule_v19 import wrap_v18_as_v19
    from coordpy.mergeable_latent_capsule_v20 import wrap_v19_as_v20
    from coordpy.mergeable_latent_capsule_v21 import wrap_v20_as_v21
    from coordpy.mergeable_latent_capsule_v22 import wrap_v21_as_v22
    from coordpy.mergeable_latent_capsule_v23 import wrap_v22_as_v23
    from coordpy.mergeable_latent_capsule_v24 import wrap_v23_as_v24
    from coordpy.mergeable_latent_capsule_v25 import wrap_v24_as_v25
    v3 = make_root_capsule_v3(
        branch_id="t", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    v = v3
    for w in (
            wrap_v3_as_v4, wrap_v4_as_v5, wrap_v5_as_v6,
            wrap_v6_as_v7, wrap_v7_as_v8, wrap_v8_as_v9,
            wrap_v9_as_v10, wrap_v10_as_v11, wrap_v11_as_v12,
            wrap_v12_as_v13, wrap_v13_as_v14, wrap_v14_as_v15,
            wrap_v15_as_v16, wrap_v16_as_v17, wrap_v17_as_v18,
            wrap_v18_as_v19):
        v = w(v)
    v = wrap_v19_as_v20(
        v, restart_repair_trajectory_chain=(),
        rejoin_pressure_chain=())
    v = wrap_v20_as_v21(
        v, replacement_repair_trajectory_chain=(),
        contradiction_chain=())
    v = wrap_v21_as_v22(
        v, compound_repair_trajectory_chain=(),
        delayed_repair_chain=())
    v = wrap_v22_as_v23(
        v, compound_chain_repair_trajectory_chain=(),
        replacement_then_rejoin_chain=())
    v = wrap_v23_as_v24(
        v, chain_then_restart_trajectory_chain=(),
        post_compound_chain_restart_chain=())
    v = wrap_v24_as_v25(
        v, post_restart_replacement_trajectory_chain=(),
        post_restart_replacement_window_chain=())
    v = wrap_v25_as_v26(
        v, long_horizon_reconstruction_trajectory_chain=(),
        reconstruction_request_window_chain=())
    v27 = wrap_v26_as_v27(
        v,
        replacement_then_restart_after_long_delay_trajectory_chain=(
            "rtrld_x",),
        replacement_then_restart_after_long_delay_window_chain=(
            "win_1",))
    w = emit_mlsc_v27_witness(v27)
    assert int(
        w.replacement_then_restart_after_long_delay_trajectory_chain_depth) >= 1


def test_w79_masc_v15_baseline_regime():
    from coordpy.multi_agent_substrate_coordinator_v15 import (
        MultiAgentSubstrateCoordinatorV15,
        W79_MASC_V15_REGIMES,
    )
    masc = MultiAgentSubstrateCoordinatorV15()
    assert len(W79_MASC_V15_REGIMES) == 19
    _, agg = masc.run_batch(seeds=[1, 2, 3], regime="baseline")
    assert float(agg.v24_beats_v23_rate) >= 0.5
    assert float(agg.tsc_v24_beats_tsc_v23_rate) >= 0.5


def test_w79_masc_v15_new_regime():
    from coordpy.multi_agent_substrate_coordinator_v15 import (
        MultiAgentSubstrateCoordinatorV15,
        W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY,
    )
    masc = MultiAgentSubstrateCoordinatorV15()
    _, agg = masc.run_batch(
        seeds=[1, 2, 3],
        regime=W79_MASC_V15_REGIME_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY)
    assert float(agg.v24_beats_v23_rate) >= 0.5
    assert float(agg.tsc_v24_beats_tsc_v23_rate) >= 0.5


def test_w79_tcc_v14():
    from coordpy.team_consensus_controller_v14 import (
        TeamConsensusControllerV14,
        W79_TC_V14_DECISIONS,
        W79_TC_V14_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PRESSURE,
        emit_team_consensus_controller_v14_witness,
    )
    tcc = TeamConsensusControllerV14()
    d = tcc.decide_v14(
        regime="long_delay_reconstruction_after_replacement_then_restart",
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        replacement_then_restart_after_long_delay_pressure=0.9,
        agent_replacement_then_restart_after_long_delay_recovery_flags=[
            1, 0, 1, 0])
    assert d["decision"] in W79_TC_V14_DECISIONS
    w = emit_team_consensus_controller_v14_witness(tcc)
    assert int(w.n_decisions_v14) >= 1


def test_w79_substrate_adapter_v24():
    from coordpy.substrate_adapter_v24 import (
        W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1,
        W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL,
        probe_all_v24_adapters,
    )
    adapters = probe_all_v24_adapters()
    assert adapters.has_v24_full()
    assert adapters.has_controlled_runtime()


def test_w79_deep_substrate_hybrid_v24():
    from coordpy.deep_substrate_hybrid_v23 import (
        DeepSubstrateHybridV23ForwardWitness,
    )
    from coordpy.deep_substrate_hybrid_v24 import (
        DeepSubstrateHybridV24, deep_substrate_hybrid_v24_forward,
    )
    h = DeepSubstrateHybridV24()
    v23_w = DeepSubstrateHybridV23ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v23.v1",
        hybrid_cid="", inner_v22_witness_cid="",
        twenty_three_way=True,
        cache_controller_v21_fired=True,
        replay_controller_v19_fired=True,
        long_horizon_reconstruction_trajectory_active=True,
        long_horizon_reconstruction_repair_active=True,
        team_consensus_controller_v13_active=True,
        long_horizon_reconstruction_trajectory_cid="lhr",
        long_horizon_reconstruction_repair_l1=1,
        long_horizon_reconstruction_pressure_gate_mean=0.5)
    w = deep_substrate_hybrid_v24_forward(
        hybrid=h, v23_witness=v23_w,
        replacement_then_restart_after_long_delay_trajectory_cid="rtrld",
        replacement_then_restart_after_long_delay_repair_l1=1,
        replacement_then_restart_after_long_delay_pressure_gate_mean=0.5,
        n_team_consensus_v14_invocations=1)
    assert w.twenty_four_way


def test_w79_hosted_boundary_v12():
    from coordpy.hosted_real_substrate_boundary_v12 import (
        build_default_hosted_real_substrate_boundary_v12,
        build_wall_report_v12,
        probe_hosted_real_substrate_boundary_v12_falsifier,
    )
    b = build_default_hosted_real_substrate_boundary_v12()
    assert len(b.blocked_axes) >= 56
    r = build_wall_report_v12(boundary=b)
    assert len(r.controlled_runtime_only_axes) >= 1
    f = probe_hosted_real_substrate_boundary_v12_falsifier(
        boundary=b,
        claimed_axis=(
            "replacement_then_restart_after_long_delay_trajectory_cid"),
        claim_satisfied_at_hosted=False)
    assert float(f.falsifier_score) == 0.0
    f2 = probe_hosted_real_substrate_boundary_v12_falsifier(
        boundary=b,
        claimed_axis=(
            "replacement_then_restart_after_long_delay_trajectory_cid"),
        claim_satisfied_at_hosted=True)
    assert float(f2.falsifier_score) == 1.0


def test_w79_handoff_v11_controlled_runtime_promotion():
    from coordpy.hosted_real_handoff_coordinator_v11 import (
        HostedRealHandoffCoordinatorV11,
        W79_HANDOFF_DECISION_CONTROLLED_RUNTIME,
    )
    from coordpy.r197_benchmark import _make_req_v11
    coord = HostedRealHandoffCoordinatorV11()
    env = coord.decide_v11(
        req_v11=_make_req_v11(
            "t", needs_controlled_runtime=True))
    assert (
        env.decision_v11
        == W79_HANDOFF_DECISION_CONTROLLED_RUNTIME)
    assert bool(env.controlled_runtime_active)


def test_w79_hosted_provider_filter_v11_drops():
    from coordpy.hosted_provider_filter_v11 import (
        HostedProviderFilterSpecV11, filter_hosted_registry_v11,
    )
    from coordpy.hosted_router_controller import (
        default_hosted_registry,
    )
    from coordpy.w78_team import W78Params
    w78p = W78Params.build_default(seed=78002)
    reg = default_hosted_registry()
    spec = HostedProviderFilterSpecV11(
        inner_v10=w78p.hosted_provider_filter_v10,
        replacement_then_restart_after_long_delay_pressure=0.9,
        replacement_then_restart_after_long_delay_pressure_floor=0.5,
        max_replacement_then_restart_after_long_delay_noise_per_provider={
            "openrouter_paid": 0.05,
            "openai_paid": 1.0})
    _, rep = filter_hosted_registry_v11(
        reg, spec,
        provider_replacement_then_restart_after_long_delay_noise={
            "openrouter_paid": 0.5,
            "openai_paid": 0.1})
    assert int(rep.get(
        "n_dropped_under_replacement_then_restart_after_long_delay",
        0)) >= 1
