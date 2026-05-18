"""W79 — controlled-runtime substrate + façade tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_w79_controlled_runtime_deterministic_forward():
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
        forward_controlled_runtime, tokenize_bytes_v79,
    )
    p = build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79("hello w79 controlled runtime")
    t1, _ = forward_controlled_runtime(
        params=p, input_token_ids=ids)
    t2, _ = forward_controlled_runtime(
        params=p, input_token_ids=ids)
    assert t1.cid() == t2.cid()


def test_w79_controlled_runtime_replay_byte_identical():
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
        measure_controlled_runtime_replay_vs_recompute,
        tokenize_bytes_v79,
    )
    p = build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79(
        "the w79 controlled runtime supports byte-stable "
        "replay from a populated KV cache")
    rep = measure_controlled_runtime_replay_vs_recompute(
        params=p, old_token_ids=ids[:10],
        new_token_ids=ids[10:])
    assert bool(rep.replay_byte_identical)
    assert float(rep.saving_ratio) > 0.0


def test_w79_controlled_runtime_substrate_axes():
    from coordpy.controlled_runtime_substrate_v1 import (
        W79_CONTROLLED_RUNTIME_AXES,
    )
    axes = W79_CONTROLLED_RUNTIME_AXES
    assert bool(axes.exposes_hidden_state)
    assert bool(axes.exposes_kv_cache)
    assert bool(axes.exposes_attention_probs)
    assert bool(axes.exposes_per_head_attention_bias)
    assert bool(axes.exposes_prefix_state_inject)
    assert bool(axes.exposes_per_layer_logits)
    assert bool(axes.exposes_replay_from_kv)


def test_w79_controlled_runtime_hidden_state_injection_changes_output():
    from coordpy.controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
        forward_controlled_runtime, tokenize_bytes_v79,
    )
    p = build_controlled_runtime_params_v1()
    ids = tokenize_bytes_v79("inject probe")
    t_base, _ = forward_controlled_runtime(
        params=p, input_token_ids=ids)
    inj = np.full(
        (len(ids), int(p.hidden_dim)),
        0.1, dtype=np.float64)
    h_inj_per_layer = [None] * int(p.n_layers)
    h_inj_per_layer[0] = inj
    t_inj, _ = forward_controlled_runtime(
        params=p, input_token_ids=ids,
        hidden_state_injections_per_layer=h_inj_per_layer)
    assert t_base.cid() != t_inj.cid()


def test_w79_local_openai_facade_chat_completion():
    from coordpy.local_openai_compatible_facade_v1 import (
        LocalOpenAIChatCompletionRequestV1,
        LocalOpenAIChatMessageV1,
        LocalOpenAICompatibleFacadeV1,
    )
    f = LocalOpenAICompatibleFacadeV1()
    req = LocalOpenAIChatCompletionRequestV1(
        model="coordpy.controlled_runtime_substrate_v1",
        messages=(
            LocalOpenAIChatMessageV1(
                role="user", content="hi"),),
        substrate_return_hidden_state=True,
        substrate_return_kv_cache_cid=True,
        substrate_return_attention_probs=True)
    r1 = f.chat_completions_create(request=req)
    r2 = f.chat_completions_create(request=req)
    assert r1.cid() == r2.cid()
    assert r1.substrate_side_channel is not None
    assert (
        r1.substrate_side_channel.cid()
        == r2.substrate_side_channel.cid())


def test_w79_facade_vs_hosted_substrate_comparison():
    from coordpy.hosted_real_substrate_boundary_v12 import (
        build_default_hosted_real_substrate_boundary_v12,
    )
    from coordpy.local_openai_compatible_facade_v1 import (
        LocalOpenAIChatCompletionRequestV1,
        LocalOpenAIChatMessageV1,
        LocalOpenAICompatibleFacadeV1,
        compare_facade_vs_hosted_substrate,
    )
    f = LocalOpenAICompatibleFacadeV1()
    req = LocalOpenAIChatCompletionRequestV1(
        model="coordpy.controlled_runtime_substrate_v1",
        messages=(
            LocalOpenAIChatMessageV1(
                role="user", content="hi"),),
        substrate_return_hidden_state=True,
        substrate_return_kv_cache_cid=True)
    r = f.chat_completions_create(request=req)
    boundary = build_default_hosted_real_substrate_boundary_v12()
    cmp = compare_facade_vs_hosted_substrate(
        facade_response=r,
        hosted_blocked_axes=tuple(boundary.blocked_axes))
    assert len(cmp.hosted_axes_blocked) >= 7
    assert len(cmp.facade_axes_exposed) >= 7
    assert not cmp.hosted_substrate_side_channel_present
