"""W76 tests — handoff coordinator V8 + cross-plane savings +
chain-then-restart falsifier + provider filter V8."""

from __future__ import annotations

from coordpy.hosted_real_handoff_coordinator import (
    HandoffRequest,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3,
)
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4,
)
from coordpy.hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5,
)
from coordpy.hosted_real_handoff_coordinator_v6 import (
    HandoffRequestV6,
)
from coordpy.hosted_real_handoff_coordinator_v7 import (
    HandoffRequestV7,
)
from coordpy.hosted_real_handoff_coordinator_v8 import (
    HandoffEnvelopeV8, HandoffRequestV8,
    HostedRealHandoffCoordinatorV8,
    W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK,
    W76_HANDOFF_DECISIONS_V8,
    hosted_real_handoff_v8_chain_then_restart_aware_savings,
    probe_hosted_real_handoff_v8_chain_then_restart_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v9 import (
    build_default_hosted_real_substrate_boundary_v9,
    probe_hosted_real_substrate_boundary_v9_falsifier,
)


def _req(*, rc: str,
         compound_chain_then_restart_pressure: float = 0.0,
         compound_chain_then_restart_trajectory_cid: str = "",
         post_compound_chain_restart_window_turns: int = 0,
         needs_text_only: bool = True,
         needs_substrate_state_access: bool = False,
         ) -> HandoffRequestV8:
    return HandoffRequestV8(
        inner_v7=HandoffRequestV7(
            inner_v6=HandoffRequestV6(
                inner_v5=HandoffRequestV5(
                    inner_v4=HandoffRequestV4(
                        inner_v3=HandoffRequestV3(
                            inner_v2=HandoffRequestV2(
                                inner_v1=HandoffRequest(
                                    request_cid=str(rc),
                                    needs_text_only=bool(
                                        needs_text_only),
                                    needs_substrate_state_access=bool(
                                        needs_substrate_state_access)),
                                visible_token_budget=256,
                                baseline_token_cost=512,
                                dominant_repair_label=0),
                            restart_pressure=0.0,
                            delayed_repair_trajectory_cid="",
                            delay_turns=0,
                            expected_substrate_trust=0.7),
                        rejoin_pressure=0.0,
                        restart_repair_trajectory_cid="",
                        rejoin_lag_turns=0,
                        expected_substrate_trust_v4=0.7),
                    replacement_pressure=0.0,
                    replacement_repair_trajectory_cid="",
                    replacement_lag_turns=0,
                    expected_substrate_trust_v5=0.7),
                compound_pressure=0.0,
                compound_repair_trajectory_cid="",
                compound_window_turns=0,
                expected_substrate_trust_v6=0.7),
            compound_chain_pressure=0.0,
            compound_chain_repair_trajectory_cid="",
            compound_chain_window_turns=0,
            expected_substrate_trust_v7=0.7),
        compound_chain_then_restart_pressure=float(
            compound_chain_then_restart_pressure),
        compound_chain_then_restart_trajectory_cid=str(
            compound_chain_then_restart_trajectory_cid),
        post_compound_chain_restart_window_turns=int(
            post_compound_chain_restart_window_turns),
        expected_substrate_trust_v8=0.7)


def test_handoff_v8_has_11_decisions() -> None:
    assert len(W76_HANDOFF_DECISIONS_V8) == 11
    assert (
        W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK
        in W76_HANDOFF_DECISIONS_V8)


def test_handoff_v8_envelope_content_addressed() -> None:
    coord = HostedRealHandoffCoordinatorV8()
    env_a = coord.decide_v8(req_v8=_req(rc="a"))
    env_b = coord.decide_v8(req_v8=_req(rc="b"))
    assert env_a.cid() != env_b.cid()


def test_handoff_v8_text_only_routes_to_plane_A() -> None:
    coord = HostedRealHandoffCoordinatorV8()
    env = coord.decide_v8(req_v8=_req(rc="t"))
    assert str(env.decision_v8) == W69_HANDOFF_DECISION_HOSTED_ONLY


def test_handoff_v8_substrate_only_routes_to_plane_B() -> None:
    coord = HostedRealHandoffCoordinatorV8()
    env = coord.decide_v8(
        req_v8=_req(
            rc="s",
            needs_text_only=False,
            needs_substrate_state_access=True))
    assert (
        str(env.decision_v8)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)


def test_handoff_v8_chain_then_restart_promotion() -> None:
    coord = HostedRealHandoffCoordinatorV8()
    env = coord.decide_v8(
        req_v8=_req(
            rc="ctr",
            compound_chain_then_restart_pressure=0.9,
            compound_chain_then_restart_trajectory_cid="x",
            post_compound_chain_restart_window_turns=4))
    assert (
        str(env.decision_v8)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert float(env.compound_chain_then_restart_alignment) == 1.0


def test_handoff_v8_chain_then_restart_fallback() -> None:
    coord = HostedRealHandoffCoordinatorV8()
    env = coord.decide_v8(
        req_v8=_req(
            rc="ctr-fb",
            compound_chain_then_restart_pressure=0.0,
            compound_chain_then_restart_trajectory_cid="x",
            post_compound_chain_restart_window_turns=4))
    assert (
        str(env.decision_v8)
        == W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK)
    assert bool(env.chain_then_restart_fallback_active)


def test_handoff_v8_cross_plane_savings_ge_85_percent() -> None:
    sav = hosted_real_handoff_v8_chain_then_restart_aware_savings(
        n_turns=100)
    assert float(sav["saving_ratio"]) >= 0.85


def test_handoff_v8_chain_then_restart_falsifier() -> None:
    coord = HostedRealHandoffCoordinatorV8()
    env_promoted = coord.decide_v8(
        req_v8=_req(
            rc="promoted",
            compound_chain_then_restart_pressure=0.9,
            compound_chain_then_restart_trajectory_cid="x",
            post_compound_chain_restart_window_turns=4))
    env_hosted = coord.decide_v8(req_v8=_req(rc="hosted-only"))
    f_h = (
        probe_hosted_real_handoff_v8_chain_then_restart_falsifier(
            envelope_v8=env_promoted, claim_satisfied=True))
    f_d = (
        probe_hosted_real_handoff_v8_chain_then_restart_falsifier(
            envelope_v8=env_hosted, claim_satisfied=True))
    assert float(f_h.falsifier_score) == 0.0
    assert float(f_d.falsifier_score) == 1.0


def test_boundary_v9_has_at_least_40_blocked_axes() -> None:
    b = build_default_hosted_real_substrate_boundary_v9()
    assert len(b.blocked_axes) >= 40


def test_boundary_v9_falsifier() -> None:
    b = build_default_hosted_real_substrate_boundary_v9()
    f_h = probe_hosted_real_substrate_boundary_v9_falsifier(
        boundary=b,
        claimed_axis=(
            "compound_chain_then_restart_trajectory_cid"),
        claim_satisfied_at_hosted=False)
    f_d = probe_hosted_real_substrate_boundary_v9_falsifier(
        boundary=b,
        claimed_axis=(
            "compound_chain_then_restart_trajectory_cid"),
        claim_satisfied_at_hosted=True)
    assert float(f_h.falsifier_score) == 0.0
    assert float(f_d.falsifier_score) == 1.0
