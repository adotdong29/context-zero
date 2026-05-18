"""W79 — learned consolidation tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_w79_learned_consolidation_reduces_loss():
    from coordpy.learned_consolidation_v1 import (
        build_learned_consolidation_head_v1,
        build_nonlinear_consolidation_dataset,
        train_learned_consolidation_head,
    )
    head = build_learned_consolidation_head_v1(seed=7)
    X, Y = build_nonlinear_consolidation_dataset(
        n_samples=80, seed=7)
    head, rep = train_learned_consolidation_head(
        head=head, train_features=X.tolist(),
        train_targets=Y.tolist())
    assert float(rep.post_loss) < float(rep.pre_loss)
    assert bool(rep.converged)


def test_w79_learned_consolidation_beats_ridge():
    from coordpy.learned_consolidation_v1 import (
        build_learned_consolidation_head_v1,
        build_nonlinear_consolidation_dataset,
        compare_learned_vs_closed_form,
        train_learned_consolidation_head,
    )
    head = build_learned_consolidation_head_v1(seed=9)
    X, Y = build_nonlinear_consolidation_dataset(
        n_samples=96, seed=9)
    head, _ = train_learned_consolidation_head(
        head=head, train_features=X.tolist(),
        train_targets=Y.tolist())
    cmp = compare_learned_vs_closed_form(
        head=head, eval_features=X.tolist(),
        eval_targets=Y.tolist())
    assert bool(cmp.learned_strictly_beats_ridge)
    assert float(cmp.learned_mse) < float(cmp.ridge_mse)


def test_w79_learned_consolidation_training_deterministic():
    from coordpy.learned_consolidation_v1 import (
        build_learned_consolidation_head_v1,
        build_nonlinear_consolidation_dataset,
        train_learned_consolidation_head,
    )
    h1 = build_learned_consolidation_head_v1(seed=11)
    X, Y = build_nonlinear_consolidation_dataset(
        n_samples=64, seed=11)
    h1, _ = train_learned_consolidation_head(
        head=h1, train_features=X.tolist(),
        train_targets=Y.tolist())
    h2 = build_learned_consolidation_head_v1(seed=11)
    X2, Y2 = build_nonlinear_consolidation_dataset(
        n_samples=64, seed=11)
    h2, _ = train_learned_consolidation_head(
        head=h2, train_features=X2.tolist(),
        train_targets=Y2.tolist())
    assert h1.cid() == h2.cid()


def test_w79_lhr_substrate_v2_learned_slots():
    from coordpy.learned_consolidation_v1 import (
        build_learned_consolidation_head_v1,
    )
    from coordpy.long_horizon_reconstruction_substrate_v2 import (
        build_default_long_horizon_reconstruction_carrier_v2,
    )
    head = build_learned_consolidation_head_v1(seed=13)
    carrier_v2 = (
        build_default_long_horizon_reconstruction_carrier_v2(
            n_events=64, seed=13, head=head))
    assert len(carrier_v2.learned_slots) == 64


def test_w79_replay_vs_recompute_v2_arbitration():
    from coordpy.learned_consolidation_v1 import (
        build_learned_consolidation_head_v1,
    )
    from coordpy.long_horizon_reconstruction_substrate_v1 import (
        LongHorizonReconstructionQuery,
    )
    from coordpy.long_horizon_reconstruction_substrate_v2 import (
        W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY,
        W79_REPLAY_VS_RECOMPUTE_DECISIONS,
        arbitrate_replay_vs_recompute_v2,
        build_default_long_horizon_reconstruction_carrier_v2,
    )
    head = build_learned_consolidation_head_v1(seed=17)
    carrier = build_default_long_horizon_reconstruction_carrier_v2(
        n_events=64, seed=17, head=head)
    q = LongHorizonReconstructionQuery(
        query_id="q", source_turn=5, current_turn=200)
    arb = arbitrate_replay_vs_recompute_v2(
        carrier_v2=carrier, query=q,
        controlled_runtime_params_cid="fake")
    assert arb.chosen in W79_REPLAY_VS_RECOMPUTE_DECISIONS
    # Default config makes substrate replay cheapest.
    assert (
        arb.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY)


def test_w79_bounded_window_v2_falsifier():
    from coordpy.bounded_window_baseline_v1 import (
        BoundedWindowQuery,
    )
    from coordpy.bounded_window_baseline_v2 import (
        build_default_bounded_window_baselines_v2,
        prove_bounded_window_insufficient_v2,
        run_bounded_window_falsifier_v2,
    )
    baselines = build_default_bounded_window_baselines_v2()
    q = BoundedWindowQuery(
        query_id="q", current_turn=300, source_turn=5,
        expected_event_cid="ev")
    _, rep = run_bounded_window_falsifier_v2(
        baselines_v2=baselines, query=q)
    assert bool(rep.all_fixed_k_failed_v2)
    assert bool(rep.cross_prompt_summary_failed)
    proof = prove_bounded_window_insufficient_v2(
        query_horizon_turns=290, baselines_v2=baselines)
    assert bool(proof.proven)
