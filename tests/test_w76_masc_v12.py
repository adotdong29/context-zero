"""W76 tests — MASC V12 strict-beats V11 / V10 / V9 across 16
regimes."""

from __future__ import annotations

from coordpy.multi_agent_substrate_coordinator_v12 import (
    MultiAgentSubstrateCoordinatorV12,
    W76_MASC_V12_POLICIES,
    W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21,
    W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21,
    W76_MASC_V12_REGIMES,
    W76_MASC_V12_REGIME_CHAIN_THEN_RESTART,
)


def test_masc_v12_has_16_regimes() -> None:
    assert len(W76_MASC_V12_REGIMES) == 16
    assert (
        W76_MASC_V12_REGIME_CHAIN_THEN_RESTART
        in W76_MASC_V12_REGIMES)
    # All W75 regimes still load-bearing.
    assert (
        "compound_repair_after_replacement_then_rejoin_under_budget"
        in W76_MASC_V12_REGIMES)


def test_masc_v12_has_2_new_policies() -> None:
    assert (
        W76_MASC_V12_POLICY_SUBSTRATE_ROUTED_V21
        in W76_MASC_V12_POLICIES)
    assert (
        W76_MASC_V12_POLICY_TEAM_SUBSTRATE_COORDINATION_V21
        in W76_MASC_V12_POLICIES)


def test_masc_v12_v21_beats_v20_across_all_regimes() -> None:
    masc = MultiAgentSubstrateCoordinatorV12()
    for regime in W76_MASC_V12_REGIMES:
        _, agg = masc.run_batch(
            seeds=[1, 2, 3, 4, 5], regime=regime,
            n_agents=5, n_turns=12,
            budget_tokens_per_turn=64,
            target_tolerance=0.1)
        assert (
            agg.v21_beats_v20_rate >= 0.5
        ), f"regime={regime} v21>v20={agg.v21_beats_v20_rate}"
        assert (
            agg.tsc_v21_beats_tsc_v20_rate >= 0.5
        ), f"regime={regime} tsc21>tsc20={agg.tsc_v21_beats_tsc_v20_rate}"


def test_masc_v12_new_regime_v21_strict_beat() -> None:
    masc = MultiAgentSubstrateCoordinatorV12()
    _, agg = masc.run_batch(
        seeds=[1, 2, 3, 4, 5],
        regime=W76_MASC_V12_REGIME_CHAIN_THEN_RESTART)
    # On the new regime V20 has no chain-then-restart-pressure
    # signal so V21 should strictly beat ≥ 50 %.
    assert agg.v21_beats_v20_rate >= 0.5
    assert agg.tsc_v21_beats_tsc_v20_rate >= 0.5


def test_masc_v12_carries_forward_w75_chain_regime() -> None:
    masc = MultiAgentSubstrateCoordinatorV12()
    _, agg = masc.run_batch(
        seeds=[1, 2, 3, 4, 5],
        regime=(
            "compound_repair_after_replacement_then_rejoin_"
            "under_budget"))
    assert agg.v21_beats_v20_rate >= 0.5
    assert agg.tsc_v21_beats_tsc_v20_rate >= 0.5
