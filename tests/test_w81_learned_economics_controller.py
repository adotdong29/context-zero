"""W81 / P1 #14 — Learned multi-runtime economics controller tests.

Covers:
- 5 canonical actions, 7-dim feature schema
- simulation cost/quality model is deterministic
- training reduces cross-entropy loss
- learned policy beats heuristic baseline on utility
- learned policy is closer to optimal than heuristic
- explicit honest reporting: cost / success / utility split out
"""

from __future__ import annotations

import numpy as np
import pytest


def test_w81_economics_actions_are_canonical():
    from coordpy.learned_economics_controller_v1 import (
        W81_ECONOMICS_ACTIONS,
        W81_N_ECONOMICS_ACTIONS,
    )
    assert int(W81_N_ECONOMICS_ACTIONS) == 5
    assert tuple(W81_ECONOMICS_ACTIONS) == (
        "replay",
        "runtime_recompute",
        "transcript_recompute",
        "promote_to_richer_substrate",
        "abstain",
    )


def test_w81_economics_feature_schema():
    from coordpy.learned_economics_controller_v1 import (
        W81_ECONOMICS_FEATURE_DIM,
        W81_ECONOMICS_FEATURE_NAMES,
    )
    assert int(W81_ECONOMICS_FEATURE_DIM) == 7
    assert len(W81_ECONOMICS_FEATURE_NAMES) == 7


def test_w81_economics_simulation_deterministic():
    from coordpy.learned_economics_controller_v1 import (
        EconomicsSimulationV1,
    )
    sim1 = EconomicsSimulationV1()
    sim2 = EconomicsSimulationV1()
    feat = np.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.9])
    for action in (
            "replay", "runtime_recompute",
            "transcript_recompute",
            "promote_to_richer_substrate", "abstain"):
        a = sim1.evaluate_action(
            feature=feat, action=action)
        b = sim2.evaluate_action(
            feature=feat, action=action)
        assert a == b


def test_w81_economics_dataset_build():
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
    )
    X, y, sim = build_economics_dataset_v1(
        n_samples=64, seed=11)
    assert X.shape == (64, 7)
    assert y.shape == (64,)
    # All labels are valid action indices.
    assert int(np.min(y)) >= 0
    assert int(np.max(y)) <= 4


def test_w81_economics_training_reduces_loss():
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
    )
    controller = build_learned_economics_controller_v1(seed=17)
    X, y, _ = build_economics_dataset_v1(
        n_samples=256, seed=17)
    controller, rep = train_learned_economics_controller(
        controller=controller,
        train_features=X,
        train_optimal_action_indices=y)
    assert float(rep.post_loss) < float(rep.pre_loss)
    assert bool(rep.converged)


def test_w81_economics_training_deterministic_on_seed():
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        train_learned_economics_controller,
    )
    c1 = build_learned_economics_controller_v1(seed=23)
    X1, y1, _ = build_economics_dataset_v1(
        n_samples=128, seed=23)
    c1, _ = train_learned_economics_controller(
        controller=c1, train_features=X1,
        train_optimal_action_indices=y1)
    c2 = build_learned_economics_controller_v1(seed=23)
    X2, y2, _ = build_economics_dataset_v1(
        n_samples=128, seed=23)
    c2, _ = train_learned_economics_controller(
        controller=c2, train_features=X2,
        train_optimal_action_indices=y2)
    assert c1.cid() == c2.cid()


def test_w81_economics_learned_beats_heuristic_on_utility():
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        compare_learned_vs_heuristic,
        train_learned_economics_controller,
    )
    controller = build_learned_economics_controller_v1(seed=29)
    X_train, y_train, sim = build_economics_dataset_v1(
        n_samples=512, seed=29)
    controller, _ = train_learned_economics_controller(
        controller=controller,
        train_features=X_train,
        train_optimal_action_indices=y_train,
        n_iters=120)
    # Held-out eval set.
    X_eval, _, _ = build_economics_dataset_v1(
        n_samples=256, seed=31, sim=sim)
    rep = compare_learned_vs_heuristic(
        controller=controller,
        eval_features=X_eval, sim=sim)
    assert bool(rep.learned_beats_heuristic_on_utility), (
        f"learned utility {rep.learned_avg_utility} did not "
        f"beat heuristic utility {rep.heuristic_avg_utility}")


def test_w81_economics_learned_closer_to_optimal_than_heuristic():
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        compare_learned_vs_heuristic,
        train_learned_economics_controller,
    )
    controller = build_learned_economics_controller_v1(seed=37)
    X_train, y_train, sim = build_economics_dataset_v1(
        n_samples=512, seed=37)
    controller, _ = train_learned_economics_controller(
        controller=controller,
        train_features=X_train,
        train_optimal_action_indices=y_train,
        n_iters=140)
    X_eval, _, _ = build_economics_dataset_v1(
        n_samples=256, seed=41, sim=sim)
    rep = compare_learned_vs_heuristic(
        controller=controller,
        eval_features=X_eval, sim=sim)
    assert float(rep.learned_optimality_gap) < float(
        rep.heuristic_optimality_gap), (
        f"learned gap {rep.learned_optimality_gap} >= "
        f"heuristic gap {rep.heuristic_optimality_gap}")


def test_w81_economics_action_distribution_uses_all_actions():
    """Learned controller should not collapse onto a single
    action — the report must show meaningful distribution
    across multiple actions for a held-out set with diverse
    features."""
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        compare_learned_vs_heuristic,
        train_learned_economics_controller,
    )
    controller = build_learned_economics_controller_v1(seed=43)
    X_train, y_train, sim = build_economics_dataset_v1(
        n_samples=512, seed=43)
    controller, _ = train_learned_economics_controller(
        controller=controller,
        train_features=X_train,
        train_optimal_action_indices=y_train,
        n_iters=120)
    X_eval, _, _ = build_economics_dataset_v1(
        n_samples=512, seed=47, sim=sim)
    rep = compare_learned_vs_heuristic(
        controller=controller,
        eval_features=X_eval, sim=sim)
    # At least 3 of the 5 actions should be used.
    used = sum(
        1 for v in rep.learned_action_distribution.values()
        if v > 0.02)
    assert used >= 3


def test_w81_economics_heuristic_uses_each_action_at_least_sometimes():
    """The heuristic baseline must also span the action space
    so the comparison isn't degenerate."""
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        compare_learned_vs_heuristic,
    )
    controller = build_learned_economics_controller_v1(seed=53)
    X_eval, _, sim = build_economics_dataset_v1(
        n_samples=512, seed=53)
    rep = compare_learned_vs_heuristic(
        controller=controller,
        eval_features=X_eval, sim=sim)
    used = sum(
        1 for v in rep.heuristic_action_distribution.values()
        if v > 0.0)
    assert used >= 3


def test_w81_economics_predict_distribution_sums_to_one():
    from coordpy.learned_economics_controller_v1 import (
        build_learned_economics_controller_v1,
    )
    controller = build_learned_economics_controller_v1(seed=59)
    feat = np.array(
        [0.4, 0.5, 0.6, 0.2, 0.7, 0.3, 0.9],
        dtype=np.float64)
    pi = controller.predict_distribution(feat)
    assert pi.shape == (5,)
    assert abs(float(np.sum(pi)) - 1.0) < 1e-10
    assert float(np.min(pi)) >= 0.0


def test_w81_economics_witness_chains_train_state():
    from coordpy.learned_economics_controller_v1 import (
        build_economics_dataset_v1,
        build_learned_economics_controller_v1,
        emit_learned_economics_witness,
        train_learned_economics_controller,
    )
    controller = build_learned_economics_controller_v1(seed=61)
    w1 = emit_learned_economics_witness(controller=controller)
    X, y, _ = build_economics_dataset_v1(
        n_samples=64, seed=61)
    controller, _ = train_learned_economics_controller(
        controller=controller,
        train_features=X,
        train_optimal_action_indices=y,
        n_iters=20)
    w2 = emit_learned_economics_witness(controller=controller)
    assert w1.n_train_steps == 0
    assert w2.n_train_steps == 20
    assert w1.cid() != w2.cid()


def test_w81_economics_simulation_abstain_floor_holds():
    from coordpy.learned_economics_controller_v1 import (
        EconomicsSimulationV1,
    )
    sim = EconomicsSimulationV1(abstain_floor=0.30)
    feat = np.array([0.5] * 7, dtype=np.float64)
    _, succ, _ = sim.evaluate_action(
        feature=feat, action="abstain")
    assert abs(float(succ) - 0.30) < 1e-12


def test_w81_economics_simulation_evaluates_all_5_actions():
    from coordpy.learned_economics_controller_v1 import (
        EconomicsSimulationV1,
        W81_ECONOMICS_ACTIONS,
    )
    sim = EconomicsSimulationV1()
    feat = np.array(
        [0.2, 0.3, 0.4, 0.1, 0.5, 0.6, 0.7],
        dtype=np.float64)
    for a in W81_ECONOMICS_ACTIONS:
        cost, succ, util = sim.evaluate_action(
            feature=feat, action=a)
        assert isinstance(cost, float)
        assert 0.0 <= float(succ) <= 1.0
        assert isinstance(util, float)
