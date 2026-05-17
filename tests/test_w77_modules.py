"""W77 tests — direct module sanity for tiny_substrate_v22, KV
bridge V22, cache controller V20, replay controller V18,
persistent V29, LHR V29, MLSC V25, consensus V23, deep substrate
hybrid V22, substrate adapter V22, MASC V13, TSC V12."""

from __future__ import annotations

import numpy as _np

from coordpy.cache_controller_v20 import (
    CacheControllerV20,
    W77_CACHE_POLICIES_V20,
    emit_cache_controller_v20_witness,
)
from coordpy.consensus_fallback_controller_v23 import (
    ConsensusFallbackControllerV23,
    W77_CONSENSUS_V23_STAGES,
    emit_consensus_v23_witness,
)
from coordpy.deep_substrate_hybrid_v21 import (
    DeepSubstrateHybridV21ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v22 import (
    DeepSubstrateHybridV22, deep_substrate_hybrid_v22_forward,
)
from coordpy.kv_bridge_v21 import KVBridgeV21Projection
from coordpy.kv_bridge_v22 import (
    KVBridgeV22Projection,
    W77_KV_V22_FINGERPRINT_DIM,
    compute_post_restart_replacement_fingerprint_v22,
    emit_kv_bridge_v22_witness,
    probe_kv_bridge_v22_post_restart_replacement_falsifier,
)
from coordpy.long_horizon_retention_v29 import (
    LongHorizonReconstructionV29Head,
    W77_DEFAULT_LHR_V29_MAX_K,
    emit_lhr_v29_witness,
)
from coordpy.mergeable_latent_capsule_v25 import (
    MergeOperatorV25,
    W77_MLSC_V25_KNOWN_ALGEBRA_SIGNATURES,
    emit_mlsc_v25_witness,
)
from coordpy.multi_agent_substrate_coordinator_v13 import (
    MultiAgentSubstrateCoordinatorV13,
    W77_MASC_V13_POLICIES,
    W77_MASC_V13_REGIMES,
    W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT,
    emit_multi_agent_substrate_coordinator_v13_witness,
)
from coordpy.persistent_latent_v29 import (
    PersistentLatentStateV29Chain,
    W77_DEFAULT_V29_N_LAYERS,
    emit_persistent_v29_witness,
)
from coordpy.replay_controller_v18 import (
    ReplayControllerV18,
    W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS,
    W77_REPLAY_REGIMES_V18,
    emit_replay_controller_v18_witness,
)
from coordpy.substrate_adapter_v22 import (
    W77_SUBSTRATE_TIER_SUBSTRATE_V22_FULL,
    probe_all_v22_adapters,
)
from coordpy.team_consensus_controller_v12 import (
    TeamConsensusControllerV12,
    W77_TC_V12_DECISIONS,
    emit_team_consensus_controller_v12_witness,
)
from coordpy.tiny_substrate_v22 import (
    W77_DEFAULT_V22_N_LAYERS, W77_REPAIR_LABELS_V22,
    build_default_tiny_substrate_v22,
    emit_tiny_substrate_v22_forward_witness,
    forward_tiny_substrate_v22,
    record_post_restart_replacement_window_v22,
    tokenize_bytes_v22,
)


def test_tiny_substrate_v22_forward() -> None:
    p = build_default_tiny_substrate_v22(seed=77100)
    ids = tokenize_bytes_v22("w77", max_len=8)
    trace, cache = forward_tiny_substrate_v22(p, ids)
    assert trace.v22_gate_score_per_layer.shape == (
        int(W77_DEFAULT_V22_N_LAYERS),)
    assert (
        cache
        .replacement_after_restart_after_compound_chain_length_per_layer
        .shape) == (int(W77_DEFAULT_V22_N_LAYERS),)
    w = emit_tiny_substrate_v22_forward_witness(trace, cache)
    assert int(w.n_layers) == int(W77_DEFAULT_V22_N_LAYERS)


def test_tiny_substrate_v22_records_pcr_window() -> None:
    p = build_default_tiny_substrate_v22(seed=77100)
    ids = tokenize_bytes_v22("w77", max_len=8)
    trace, cache = forward_tiny_substrate_v22(
        p, ids, post_restart_replacement_pressure=0.9)
    cid_before = str(
        cache
        .replacement_after_restart_after_compound_chain_trajectory_cid)
    record_post_restart_replacement_window_v22(
        cache, chain_then_restart_repair_turn=18,
        replacement_turn=22,
        post_restart_replacement_window_turns=10)
    trace, cache = forward_tiny_substrate_v22(
        p, ids, v22_kv_cache=cache,
        post_restart_replacement_pressure=0.9)
    cid_after = str(
        cache
        .replacement_after_restart_after_compound_chain_trajectory_cid)
    assert cid_before != cid_after


def test_kv_bridge_v22_falsifier_honest() -> None:
    f = probe_kv_bridge_v22_post_restart_replacement_falsifier(
        post_restart_replacement_pressure_flag=1)
    assert float(f.falsifier_score) == 0.0


def test_kv_bridge_v22_fingerprint_dim() -> None:
    fp = compute_post_restart_replacement_fingerprint_v22(
        role="x",
        compound_chain_then_restart_trajectory_cid="cid",
        replacement_after_restart_after_compound_chain_trajectory_cid="pcr_cid")
    assert len(fp) == W77_KV_V22_FINGERPRINT_DIM


def test_cache_controller_v20_policies() -> None:
    assert (
        "seventeen_objective_v20" in W77_CACHE_POLICIES_V20)
    cc = CacheControllerV20.init(fit_seed=77100)
    w = emit_cache_controller_v20_witness(cc)
    assert int(w.n_objectives_trained) == 0


def test_replay_controller_v18_counts() -> None:
    assert len(W77_REPLAY_REGIMES_V18) == 25
    assert (
        len(W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS)
        == 15)
    rc = ReplayControllerV18.init()
    w = emit_replay_controller_v18_witness(rc)
    assert int(w.n_per_role_per_regime_heads) == 0


def test_persistent_v29_layers() -> None:
    assert int(W77_DEFAULT_V29_N_LAYERS) == 28
    chain = PersistentLatentStateV29Chain.empty()
    w = emit_persistent_v29_witness(chain)
    assert int(w.n_layers) == 28


def test_lhr_v29_max_k() -> None:
    assert int(W77_DEFAULT_LHR_V29_MAX_K) == 1024
    head = LongHorizonReconstructionV29Head.init(seed=77100)
    w = emit_lhr_v29_witness(head, carrier=[0.1] * 6, k=8)
    assert int(w.max_k) == 1024
    assert int(w.n_heads) == 28


def test_mlsc_v25_known_signatures() -> None:
    assert len(W77_MLSC_V25_KNOWN_ALGEBRA_SIGNATURES) >= 3


def test_consensus_v23_stages() -> None:
    assert len(W77_CONSENSUS_V23_STAGES) == 40
    c = ConsensusFallbackControllerV23.init()
    w = emit_consensus_v23_witness(c)
    assert int(w.n_decisions) == 0


def test_deep_substrate_hybrid_v22() -> None:
    h = DeepSubstrateHybridV22()
    v21_w = DeepSubstrateHybridV21ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v21.v1",
        hybrid_cid="",
        inner_v20_witness_cid="",
        twenty_one_way=True,
        cache_controller_v19_fired=True,
        replay_controller_v17_fired=True,
        chain_then_restart_trajectory_active=True,
        chain_then_restart_repair_active=True,
        team_consensus_controller_v11_active=True,
        chain_then_restart_trajectory_cid="cid",
        chain_then_restart_repair_l1=1,
        chain_then_restart_pressure_gate_mean=0.5,
    )
    # With no V20 cache controller / V18 replay controller, the
    # twenty-two-way flag is False even when V21 fired.
    w = deep_substrate_hybrid_v22_forward(
        hybrid=h, v21_witness=v21_w,
        post_restart_replacement_trajectory_cid="cid",
        post_restart_replacement_repair_l1=1,
        post_restart_replacement_pressure_gate_mean=0.5,
        n_team_consensus_v12_invocations=1)
    assert not bool(w.twenty_two_way)


def test_substrate_adapter_v22_has_v22_full() -> None:
    matrix = probe_all_v22_adapters()
    assert bool(matrix.has_v22_full())


def test_masc_v13_regimes() -> None:
    assert len(W77_MASC_V13_REGIMES) == 17
    assert (
        W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT
        in W77_MASC_V13_REGIMES)
    assert len(W77_MASC_V13_POLICIES) >= 5


def test_masc_v13_v22_beats_v21_on_pcr_regime() -> None:
    masc = MultiAgentSubstrateCoordinatorV13()
    _, agg = masc.run_batch(
        seeds=[1, 2, 3, 4, 5],
        regime=W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT)
    assert float(agg.v22_beats_v21_rate) >= 0.5
    assert float(agg.tsc_v22_beats_tsc_v21_rate) >= 0.5


def test_tcc_v12_arbiter() -> None:
    t = TeamConsensusControllerV12()
    d = t.decide_v12(
        regime=(
            "replacement_after_restart_after_compound_chain_"
            "repair_under_budget"),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        post_restart_replacement_trajectory_cid="cid",
        post_restart_replacement_window_turns=4,
        agent_post_restart_replacement_absorption_scores=[
            0.97, 0.6, 0.5, 0.4])
    assert (
        str(d.get("decision"))
        == "post_restart_replacement_trajectory_arbiter")
    w = emit_team_consensus_controller_v12_witness(t)
    assert int(w.n_decisions_v12) == 1
    assert (
        "post_restart_replacement_pressure_arbiter"
        in W77_TC_V12_DECISIONS)
