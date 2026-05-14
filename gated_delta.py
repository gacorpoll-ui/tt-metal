# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Naive Gated Delta Network (GDN) recurrence in ttnn.

Implements the same single-step recurrence as ``ref_recurrence_single_step`` in
``test_gdn_kernel.py`` using elementwise ops and batched ``ttnn.matmul`` (no
custom fused kernel).
"""

from __future__ import annotations

import ttnn


def gdn_recurrence_fused_inplace(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g: ttnn.Tensor,
    beta: ttnn.Tensor,
    state: ttnn.Tensor,
    output: ttnn.Tensor,
    num_cores: int | None = None,
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

    g_exp = ttnn.exp(g)
    new_state = ttnn.multiply(state, g_exp)

    kv_mem = ttnn.matmul(k, new_state)
    delta = ttnn.multiply(beta, ttnn.subtract(v, kv_mem))
    batch, seq, Dk = k.shape
    k_t = ttnn.reshape(k, (batch, Dk, seq))
    new_state = ttnn.add(new_state, ttnn.matmul(k_t, delta))

    ttnn.matmul(q, new_state, optional_output_tensor=output)
    ttnn.assign(new_state, state)
