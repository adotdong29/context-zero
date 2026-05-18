"""W81 / P1 #19 — Differentiable memory substrate tests.

Covers:
- forward shape and slot accumulation
- determinism on seed
- training reduces loss
- V1 (slot memory) beats V2 (single recurrent state) on
  content-addressed recall
- V1 beats ridge on content-addressed recall
- compressed-snapshot CID is content-addressed
- witness round-trips
"""

from __future__ import annotations

import numpy as np
import pytest


def _build(*, seed=11):
    from coordpy.differentiable_memory_substrate_v1 import (
        build_differentiable_memory_substrate_v1,
    )
    return build_differentiable_memory_substrate_v1(seed=int(seed))


def _dataset(*, n=12, seed=11):
    from coordpy.differentiable_memory_substrate_v1 import (
        build_content_addressed_recall_dataset_v1,
    )
    return build_content_addressed_recall_dataset_v1(
        n_sequences=int(n), seed=int(seed))


def test_w81_diffmem_forward_shapes():
    mod = _build(seed=7)
    X, Y = _dataset(n=2, seed=7)
    H, slots, R, Yhat = mod.forward_sequence(X[0])
    T = int(X[0].shape[0])
    assert H.shape == (T, mod.hidden_dim)
    assert slots.shape == (mod.K_slots, mod.memory_dim)
    assert R.shape == (T, mod.memory_dim)
    assert Yhat.shape == (T, mod.output_dim)


def test_w81_diffmem_forward_deterministic():
    mod = _build(seed=13)
    X, _ = _dataset(n=1, seed=13)
    a = mod.forward_sequence(X[0])
    b = mod.forward_sequence(X[0])
    for ai, bi in zip(a, b):
        assert np.allclose(ai, bi)


def test_w81_diffmem_slot_accumulation_nonzero():
    """Slots must actually receive writes (otherwise the slot
    machinery is dead)."""
    mod = _build(seed=17)
    X, _ = _dataset(n=1, seed=17)
    _, slots, _, _ = mod.forward_sequence(X[0])
    assert float(np.max(np.abs(slots))) > 1e-3


def test_w81_diffmem_training_reduces_loss():
    from coordpy.differentiable_memory_substrate_v1 import (
        train_differentiable_memory_substrate,
    )
    mod = _build(seed=19)
    X, Y = _dataset(n=16, seed=19)
    mod, rep = train_differentiable_memory_substrate(
        module=mod,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=80)
    assert float(rep.post_loss) < float(rep.pre_loss)
    assert bool(rep.converged)


def test_w81_diffmem_training_deterministic_on_seed():
    from coordpy.differentiable_memory_substrate_v1 import (
        train_differentiable_memory_substrate,
    )
    m1 = _build(seed=23)
    X1, Y1 = _dataset(n=6, seed=23)
    m1, _ = train_differentiable_memory_substrate(
        module=m1,
        train_sequences=X1.tolist(),
        train_targets=Y1.tolist(),
        n_iters=20)
    m2 = _build(seed=23)
    X2, Y2 = _dataset(n=6, seed=23)
    m2, _ = train_differentiable_memory_substrate(
        module=m2,
        train_sequences=X2.tolist(),
        train_targets=Y2.tolist(),
        n_iters=20)
    assert m1.cid() == m2.cid()


def test_w81_diffmem_compressed_snapshot_cid_stable():
    """Same input + same module -> same compressed slots CID."""
    mod = _build(seed=29)
    X, _ = _dataset(n=1, seed=29)
    cid_a = mod.compressed_snapshot_cid(X=X[0])
    cid_b = mod.compressed_snapshot_cid(X=X[0])
    assert cid_a == cid_b
    assert len(cid_a) == 64


def test_w81_diffmem_compressed_snapshot_cid_changes_on_input():
    """Different inputs -> different snapshot CIDs."""
    mod = _build(seed=31)
    X, _ = _dataset(n=2, seed=31)
    cid_a = mod.compressed_snapshot_cid(X=X[0])
    cid_b = mod.compressed_snapshot_cid(X=X[1])
    assert cid_a != cid_b


def test_w81_diffmem_v1_beats_v2_on_content_addressed_recall():
    """The load-bearing claim: V1 slot memory beats V2 single
    recurrent state on content-addressed recall."""
    from coordpy.differentiable_memory_substrate_v1 import (
        compare_differentiable_memory_v1_vs_v2_and_ridge,
        train_differentiable_memory_substrate,
    )
    mod = _build(seed=37)
    X, Y = _dataset(n=24, seed=37)
    mod, _ = train_differentiable_memory_substrate(
        module=mod,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=120)
    rep = compare_differentiable_memory_v1_vs_v2_and_ridge(
        v1_module=mod,
        eval_sequences=X.tolist(),
        eval_targets=Y.tolist())
    assert bool(rep.v1_beats_v2), (
        f"V1 mse={rep.v1_mse} did not beat V2 mse={rep.v2_mse}")


def test_w81_diffmem_v1_beats_ridge_on_recall():
    """V1 must also beat closed-form pointwise ridge."""
    from coordpy.differentiable_memory_substrate_v1 import (
        compare_differentiable_memory_v1_vs_v2_and_ridge,
        train_differentiable_memory_substrate,
    )
    mod = _build(seed=41)
    X, Y = _dataset(n=24, seed=41)
    mod, _ = train_differentiable_memory_substrate(
        module=mod,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=120)
    rep = compare_differentiable_memory_v1_vs_v2_and_ridge(
        v1_module=mod,
        eval_sequences=X.tolist(),
        eval_targets=Y.tolist())
    assert bool(rep.v1_beats_ridge), (
        f"V1 mse={rep.v1_mse} did not beat "
        f"ridge mse={rep.ridge_mse}")


def test_w81_diffmem_dataset_marker_query_structure():
    """Sanity check the dataset has the right structure."""
    X, Y = _dataset(n=8, seed=47)
    # At least one timestep per sequence has a nonzero target.
    for i in range(int(X.shape[0])):
        assert float(np.max(np.abs(Y[i]))) > 1e-3


def test_w81_diffmem_witness_chains_state():
    from coordpy.differentiable_memory_substrate_v1 import (
        emit_differentiable_memory_witness_v1,
        train_differentiable_memory_substrate,
    )
    mod = _build(seed=53)
    w1 = emit_differentiable_memory_witness_v1(module=mod)
    X, Y = _dataset(n=4, seed=53)
    mod, _ = train_differentiable_memory_substrate(
        module=mod,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=10)
    w2 = emit_differentiable_memory_witness_v1(module=mod)
    assert w1.n_train_steps == 0
    assert w2.n_train_steps == 10
    assert w1.cid() != w2.cid()
