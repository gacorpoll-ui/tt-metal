# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Naive Gated Delta Network (GDN) recurrence in ttnn.

Implements the same single-step recurrence as ``ref_recurrence_single_step`` in
``test_gdn_kernel.py`` using elementwise ops and batched ``ttnn.matmul`` (no
custom fused kernel).
"""

from __future__ import annotations

import torch
import ttnn

compare = True

from models.common.utility_functions import comp_pcc


def compare_or_save(name: str, a: ttnn.Tensor):
    return
    if compare:
        ref = torch.load(name + ".pt")
        actual = ttnn.to_torch(a)
        pcc = comp_pcc(ref, actual)
        print(f"PCC of {name} is {pcc}")
    else:
        torch.save(ttnn.to_torch(a), name + ".pt")


def gdn_recurrence_fused_inplace(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
    output: ttnn.Tensor | None = None,
    num_cores: int | None = None,
    iter: int = 0,
) -> None:
    """One GDN decode step: decay state, apply delta, write output.

    Shapes (batch = number of independent heads / pairs):

    - ``q``: ``[batch, 1, Dk]``
    - ``k``: ``[batch, 1, Dk]``
    - ``v``: ``[batch, 1, Dv]``
    - ``g``: ``[batch, 1, 1]`` — log-space decay (typically negative)
    - ``beta``: ``[batch, 1, 1]``
    - ``state``: ``[batch, Dk, Dv]`` — updated in place
    - ``output``: ``[batch, 1, Dv]`` — written in place

    ``num_cores`` is ignored; it is kept for API compatibility with a future
    fused kernel.
    """
    _ = num_cores
    batch, num_heads, seq, Dk = k.shape
    Dv = v.shape[-1]

    num_pairs = batch * num_heads
    if not compare:
        q = ttnn.reshape(q, (num_pairs, seq, Dk))
        k = ttnn.reshape(k, (num_pairs, seq, Dk))
        v = ttnn.reshape(v, (num_pairs, seq, Dv))
        g = ttnn.reshape(g, (num_pairs, seq, 1))
        beta = ttnn.reshape(beta, (num_pairs, seq, 1))
        state = ttnn.reshape(state, (num_pairs, Dk, Dv))
        if output is not None:
            output = ttnn.reshape(output, (num_pairs, seq, Dv))
        k_t = ttnn.permute(k, (0, 2, 1))
        eye_host = torch.eye(Dk, dtype=torch.float32).unsqueeze(0).expand(num_pairs, Dk, Dk).contiguous()
    else:
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        eye_host = torch.eye(Dk, dtype=torch.float32).unsqueeze(0).expand(batch, num_heads, Dk, Dk).contiguous()

    assert seq == 1
    g_exp = ttnn.exp(g)
    decayed = ttnn.multiply(state, g_exp)
    # new_state = (I - delta) @ decayed + beta * (k_t @ v),  delta = beta * (k_t @ k)
    kk = ttnn.matmul(k_t, k)
    delta = ttnn.multiply(beta, kk)
    identity = ttnn.from_torch(
        eye_host,
        dtype=decayed.dtype,
        layout=decayed.layout,
        device=decayed.device(),
    )
    factor = ttnn.subtract(identity, delta)
    projected = ttnn.matmul(factor, decayed)
    bktv = ttnn.multiply(beta, ttnn.matmul(k_t, v))
    new_state = ttnn.add(projected, bktv)

    compare_or_save(f"new_state_{iter}", new_state)
    compare_or_save(f"projected_{iter}", projected)
    compare_or_save(f"bktv_{iter}", bktv)
    compare_or_save(f"factor_{iter}", factor)
    compare_or_save(f"decayed_{iter}", decayed)
    compare_or_save(f"delta_{iter}", delta)
    compare_or_save(f"kk_{iter}", kk)
    compare_or_save(f"g_exp_{iter}", g_exp)
    ttnn.assign(new_state, state)
    output = ttnn.matmul(q, new_state)
    compare_or_save(f"output_{iter}", output)
    return output


def chunked_gdn_recurrence_fused_inplace(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
    output: ttnn.Tensor,
    num_cores: int | None = None,
) -> None:
    """Run ``gdn_recurrence_fused_inplace`` for each time index along ``seq``.

    Shapes (batch = number of independent heads / pairs):

    - ``q``, ``k``: ``[batch, seq, 1, Dk]``
    - ``v``, ``output``: ``[batch, seq, 1, Dv]``
    - ``g``, ``beta``: ``[batch, seq, 1, 1]``
    - ``state``: ``[batch, Dk, Dv]`` — updated in place across the loop

    ``num_cores`` is forwarded for API compatibility; the naive op ignores it.
    """
    batch, seq, _, dk = k.shape
    dv = v.shape[-1]
    outputs_list = []
    q_ts = [ttnn.squeeze(x, 1) for x in ttnn.split(q, 1, dim=1)]
    k_ts = [ttnn.squeeze(x, 1) for x in ttnn.split(k, 1, dim=1)]
    v_ts = [ttnn.squeeze(x, 1) for x in ttnn.split(v, 1, dim=1)]
    g_ts = [ttnn.squeeze(x, 1) for x in ttnn.split(g, 1, dim=1)]
    beta_ts = [ttnn.squeeze(x, 1) for x in ttnn.split(beta, 1, dim=1)]
    for t in range(seq):
        this_output = gdn_recurrence_fused_inplace(q_ts[t], k_ts[t], v_ts[t], g_ts[t], beta_ts[t], state)
        outputs_list.append(ttnn.to_torch(this_output))
    final_output = torch.stack(outputs_list, dim=1).squeeze(2).unsqueeze(0)
    return final_output
