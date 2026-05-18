"""W79 — team orchestrator tests."""

from __future__ import annotations

import pytest


def test_w79_team_trivial_passthrough():
    from coordpy.w79_team import (
        W79Params, W79Team, verify_w79_handoff,
    )
    params = W79Params.build_trivial()
    team = W79Team(params=params)
    env = team.run_team_turn(w78_outer_cid="abc123")
    assert env.w78_outer_cid == "abc123"
    assert env.substrate_v24_used is False
    assert env.controlled_runtime_used is False
    ok, fails = verify_w79_handoff(env, params, "abc123")
    assert ok, fails


def test_w79_team_default_build_and_run():
    from coordpy.w79_team import build_default_w79_team, verify_w79_handoff
    team = build_default_w79_team()
    env = team.run_team_turn(w78_outer_cid="fake_w78_cid")
    assert env.substrate_v24_used is True
    assert env.controlled_runtime_used is True
    assert env.twenty_four_way_used is True
    assert bool(env.learned_consolidation_beats_ridge)
    assert bool(env.facade_substrate_side_channel_emitted)
    assert bool(env.controlled_runtime_replay_byte_identical)
    assert bool(env.bounded_window_v2_all_fixed_k_failed)
    assert float(env.long_horizon_reconstruction_v2_success_rate) >= 0.95
    ok, fails = verify_w79_handoff(
        env, team.params, "fake_w78_cid")
    assert ok, fails


def test_w79_team_envelope_verify_drift():
    from coordpy.w79_team import build_default_w79_team, verify_w79_handoff
    team = build_default_w79_team()
    env = team.run_team_turn(w78_outer_cid="abc")
    ok, fails = verify_w79_handoff(env, team.params, "xyz")
    assert not ok
    assert "w79_outer_envelope_w78_outer_cid_drift" in fails


def test_w79_team_carries_w78_outer_byte_for_byte():
    """Trivial-passthrough preserves the W78 outer envelope CID
    byte-for-byte."""
    from coordpy.w79_team import W79Params, W79Team
    params = W79Params.build_trivial()
    team = W79Team(params=params)
    env = team.run_team_turn(
        w78_outer_cid="W78_outer_CID_under_test")
    assert (
        env.w78_outer_cid
        == "W78_outer_CID_under_test")


def test_w79_team_masc_rates_baseline_regime():
    from coordpy.w79_team import build_default_w79_team
    team = build_default_w79_team()
    env = team.run_team_turn(w78_outer_cid="cid")
    assert float(env.masc_v15_v24_beats_v23_rate) >= 0.5
    assert (
        float(env.masc_v15_tsc_v24_beats_tsc_v23_rate) >= 0.5)
