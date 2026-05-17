"""W76 tests — focused module CID + witness stability."""

from __future__ import annotations

from coordpy.cache_controller_v19 import (
    CacheControllerV19,
    W76_CACHE_POLICIES_V19,
    emit_cache_controller_v19_witness,
)
from coordpy.consensus_fallback_controller_v22 import (
    ConsensusFallbackControllerV22,
    W76_CONSENSUS_V22_STAGES,
    emit_consensus_v22_witness,
)
from coordpy.deep_substrate_hybrid_v20 import (
    DeepSubstrateHybridV20ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v21 import (
    DeepSubstrateHybridV21, deep_substrate_hybrid_v21_forward,
)
from coordpy.kv_bridge_v21 import (
    W76_KV_V21_FINGERPRINT_DIM,
    compute_chain_then_restart_fingerprint_v21,
)
from coordpy.long_horizon_retention_v28 import (
    LongHorizonReconstructionV28Head,
    W76_DEFAULT_LHR_V28_MAX_K,
    emit_lhr_v28_witness,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
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
from coordpy.mergeable_latent_capsule_v24 import (
    MergeOperatorV24,
    W76_MLSC_V24_SCHEMA_VERSION,
    emit_mlsc_v24_witness, wrap_v23_as_v24,
)
from coordpy.persistent_latent_v28 import (
    PersistentLatentStateV28Chain,
    W76_DEFAULT_V28_N_LAYERS,
    emit_persistent_v28_witness,
)
from coordpy.replay_controller_v17 import (
    ReplayControllerV17,
    W76_REPLAY_REGIMES_V17,
    W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS,
    emit_replay_controller_v17_witness,
)
from coordpy.substrate_adapter_v21 import (
    W76_SUBSTRATE_TIER_SUBSTRATE_V21_FULL,
    W76_SUBSTRATE_V21_CAPABILITY_AXES,
    W76_SUBSTRATE_V21_NEW_AXES,
    probe_all_v21_adapters,
)
from coordpy.team_consensus_controller_v11 import (
    TeamConsensusControllerV11,
    W76_TC_V11_DECISIONS,
    emit_team_consensus_controller_v11_witness,
)


def test_kv_bridge_v21_fingerprint_dim_is_140() -> None:
    assert W76_KV_V21_FINGERPRINT_DIM == 140
    fp = compute_chain_then_restart_fingerprint_v21(
        role="r", repair_trajectory_cid="x",
        delayed_repair_trajectory_cid="x",
        restart_repair_trajectory_cid="x",
        replacement_repair_trajectory_cid="x",
        compound_repair_trajectory_cid="x",
        compound_chain_repair_trajectory_cid="x",
        compound_chain_then_restart_trajectory_cid="x")
    assert len(fp) == 140


def test_cache_controller_v19_policies() -> None:
    cc = CacheControllerV19.init()
    assert "composite_v19" in cc.policy
    assert len(W76_CACHE_POLICIES_V19) >= 4
    w = emit_cache_controller_v19_witness(cc)
    assert int(w.n_objectives_trained) == 0


def test_replay_controller_v17_regimes_24() -> None:
    rc = ReplayControllerV17.init()
    assert len(W76_REPLAY_REGIMES_V17) == 24
    assert (
        "restart_after_compound_chain_repair_regime"
        in W76_REPLAY_REGIMES_V17)
    assert len(W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS) == 14
    w = emit_replay_controller_v17_witness(rc)
    assert bool(
        w.chain_then_restart_aware_routing_head_trained) is False


def test_persistent_v28_n_layers_is_27() -> None:
    assert W76_DEFAULT_V28_N_LAYERS == 27
    ch = PersistentLatentStateV28Chain.empty()
    w = emit_persistent_v28_witness(ch)
    assert int(w.n_layers) == 27


def test_lhr_v28_max_k_is_960() -> None:
    assert W76_DEFAULT_LHR_V28_MAX_K == 960
    h = LongHorizonReconstructionV28Head.init()
    w = emit_lhr_v28_witness(
        h, carrier=[0.1] * 6, k=16,
        chain_then_restart_indicator=[0.5] * 8)
    assert int(w.n_heads) == 27
    assert int(w.max_k) == 960


def test_mlsc_v24_witness() -> None:
    v3 = make_root_capsule_v3(
        branch_id="t", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    v = v3
    for fn in [
        wrap_v3_as_v4, wrap_v4_as_v5, wrap_v5_as_v6,
        wrap_v6_as_v7, wrap_v7_as_v8, wrap_v8_as_v9,
        wrap_v9_as_v10, wrap_v10_as_v11, wrap_v11_as_v12,
        wrap_v12_as_v13, wrap_v13_as_v14, wrap_v14_as_v15,
        wrap_v15_as_v16, wrap_v16_as_v17, wrap_v17_as_v18,
        wrap_v18_as_v19, wrap_v19_as_v20, wrap_v20_as_v21,
        wrap_v21_as_v22, wrap_v22_as_v23,
    ]:
        v = fn(v)
    v24 = wrap_v23_as_v24(
        v, chain_then_restart_trajectory_chain=("ctr1",),
        post_compound_chain_restart_chain=("pccr1",))
    w = emit_mlsc_v24_witness(v24)
    assert w.schema == W76_MLSC_V24_SCHEMA_VERSION
    assert int(w.chain_then_restart_trajectory_chain_depth) == 1
    assert int(w.post_compound_chain_restart_chain_depth) == 1


def test_consensus_v22_stages_ge_38() -> None:
    assert len(W76_CONSENSUS_V22_STAGES) >= 38
    c = ConsensusFallbackControllerV22.init()
    w = emit_consensus_v22_witness(c)
    assert len(w.stages) >= 38


def test_tcc_v11_decisions() -> None:
    tcc = TeamConsensusControllerV11()
    assert "chain_then_restart_pressure_arbiter" in (
        W76_TC_V11_DECISIONS)
    assert (
        "post_compound_chain_restart_after_rtr_arbiter"
        in W76_TC_V11_DECISIONS)
    w = emit_team_consensus_controller_v11_witness(tcc)
    assert int(w.n_decisions_v11) == 0


def test_substrate_adapter_v21_full() -> None:
    matrix = probe_all_v21_adapters()
    assert matrix.has_v21_full()
    assert (
        "compound_chain_then_restart_trajectory_cid"
        in W76_SUBSTRATE_V21_NEW_AXES)
    # Capability axes superset of V20.
    assert len(W76_SUBSTRATE_V21_CAPABILITY_AXES) > 50


def test_deep_substrate_hybrid_v21_twenty_one_way() -> None:
    cc = CacheControllerV19.init()
    rc = ReplayControllerV17.init()
    # Need at least one fitted head + one per-role-per-regime
    # head to fire.
    import numpy as _np
    from coordpy.cache_controller_v19 import (
        fit_sixteen_objective_ridge_v19,
    )
    from coordpy.replay_controller_v17 import (
        fit_replay_controller_v17_per_role,
        fit_replay_v17_chain_then_restart_aware_routing_head,
    )
    from coordpy.replay_controller import ReplayCandidate
    rng = _np.random.default_rng(76900)
    X = rng.standard_normal((10, 4))
    cc, _ = fit_sixteen_objective_ridge_v19(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(X[:, 0] * 0.3).tolist(),
        target_team_failure_recovery=(X[:, 2] * 0.4).tolist(),
        target_branch_merge=(X[:, 0] * 0.2).tolist(),
        target_partial_contradiction=(X[:, 1] * 0.3).tolist(),
        target_multi_branch_rejoin=(X[:, 0] * 0.5).tolist(),
        target_budget_primary=(X[:, 0] * 0.2).tolist(),
        target_restart_dominance=(X[:, 3] * 0.4).tolist(),
        target_delayed_rejoin_after_restart=(
            X[:, 1] * 0.3).tolist(),
        target_replacement_after_ctr=(X[:, 0] * 0.3).tolist(),
        target_compound_repair=(X[:, 0] * 0.25).tolist(),
        target_compound_chain_repair=(X[:, 0] * 0.2).tolist(),
        target_chain_then_restart_repair=(
            X[:, 0] * 0.15).tolist())
    cands = {
        r: [ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0)]
        for r in W76_REPLAY_REGIMES_V17}
    decs = {
        r: ["choose_reuse"] for r in W76_REPLAY_REGIMES_V17}
    rc, _ = fit_replay_controller_v17_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    X_team = rng.standard_normal((56, 10))
    labs = [
        W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS[
            i % len(W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS)]
        for i in range(56)]
    rc, _ = fit_replay_v17_chain_then_restart_aware_routing_head(
        controller=rc, train_team_features=X_team.tolist(),
        train_routing_labels=labs)
    hybrid = DeepSubstrateHybridV21()
    v20_w = DeepSubstrateHybridV20ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v20.v1",
        hybrid_cid="x", inner_v19_witness_cid="x",
        twenty_way=True, cache_controller_v18_fired=True,
        replay_controller_v16_fired=True,
        compound_chain_repair_trajectory_active=True,
        compound_chain_repair_active=True,
        team_consensus_controller_v10_active=True,
        compound_chain_repair_trajectory_cid="x",
        compound_chain_repair_l1=5,
        compound_chain_pressure_gate_mean=0.5)
    w = deep_substrate_hybrid_v21_forward(
        hybrid=hybrid, v20_witness=v20_w,
        cache_controller_v19=cc, replay_controller_v17=rc,
        chain_then_restart_trajectory_cid="ctr",
        chain_then_restart_repair_l1=4,
        chain_then_restart_pressure_gate_mean=0.6,
        n_team_consensus_v11_invocations=1)
    assert bool(w.twenty_one_way)
