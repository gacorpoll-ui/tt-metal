# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
GDN recurrence correctness test.

Reference: ``torch_recurrent_gated_delta_rule`` from Hugging Face ``transformers``
``modular_qwen3_next.py`` (Qwen3-Next linear attention recurrence).

https://github.com/huggingface/transformers/blob/ddb841f48888e3fcf50c3f2a570ac9774aa7373c/src/transformers/models/qwen3_next/modular_qwen3_next.py#L294

Compares a naive ttnn implementation (``gated_delta.gdn_recurrence_fused_inplace``)
over sequence length ``SEQ_LEN`` (64) against that reference. No conv1d or projections.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.unit_tests.gated_delta import chunked_gdn_recurrence_fused_inplace, gdn_recurrence_fused_inplace
from models.common.utility_functions import comp_pcc

# Sequence length for this test (prefill-style recurrence along time).
SEQ_LEN = 64


# --- Reference (aligned with HF modular_qwen3_next.py, linked above) -----------------


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """Match FLA / Qwen3-Next ``l2norm``."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    """Port of HF ``torch_recurrent_gated_delta_rule`` (same tensor semantics)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def preprocess_like_hf(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same preprocessing as inside ``torch_recurrent_gated_delta_rule`` before the time loop."""
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    return query, key, value, beta, g


def pearson_correlation_coefficient(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between the flattened elements of ``a`` and ``b``."""
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("num_heads", [10, 32, 384])
def test_gdn_kernel_correctness(mesh_device, reset_seeds, num_heads):
    """Match HF recurrent GDN over ``SEQ_LEN`` (64) tokens vs naive ttnn stepping."""
    device = mesh_device
    batch_size = 2
    Dk, Dv = 128, 128
    num_cores = min(10, batch_size * num_heads)

    logger.info(f"Testing GDN: batch={batch_size}, num_heads={num_heads}, seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}")

    torch.manual_seed(432)
    query = torch.randn(batch_size, num_heads, SEQ_LEN, Dk, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, SEQ_LEN, Dk, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, SEQ_LEN, Dv, dtype=torch.float32)
    g = torch.randn(batch_size, num_heads, SEQ_LEN, dtype=torch.float32) * 0.5 - 1.0
    beta = torch.randn(batch_size, num_heads, SEQ_LEN, dtype=torch.float32)

    out_ref, state_ref = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # out_ref: [B, H, S, Dv] (same layout as ``core_attn_out`` in the reference loop)
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    q_f, k_f, v_f, beta_f, g_f = preprocess_like_hf(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)
    # Shapes [B, H, S, *]
    num_pairs = batch_size * num_heads

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    state_tt = ttnn.from_torch(
        torch.zeros(batch_size, num_heads, Dk, Dv, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output_tt = ttnn.from_torch(
        torch.zeros(batch_size, num_heads, 1, Dv, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    q_tt = to_tt(q_f)
    k_tt = to_tt(k_f)
    v_tt = to_tt(v_f)
    g_tt = to_tt(g_f)
    beta_tt = to_tt(beta_f)

    step_outputs: list[torch.Tensor] = []
    for t in range(SEQ_LEN):
        q_t = q_tt[:, :, t : t + 1]
        k_t = k_tt[:, :, t : t + 1]
        v_t = v_tt[:, :, t : t + 1]
        g_t = g_tt[:, :, t : t + 1]
        beta_t = beta_tt[:, :, t : t + 1]

        output_tt = gdn_recurrence_fused_inplace(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            state_tt,
            num_cores=num_cores,
            iter=t,
        )
        step_outputs.append(ttnn.to_torch(output_tt).float().squeeze(1).clone())

    out_tt_stacked = torch.stack(step_outputs, dim=2)  # [BH, S, Dv]
    out_tt_cpu = out_tt_stacked.reshape(batch_size, num_heads, SEQ_LEN, Dv).contiguous()
    state_tt_cpu = ttnn.to_torch(state_tt).float()

    logger.info(f"ttnn output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    out_ref_f = out_ref.float()
    out_diff = (out_ref_f - out_tt_cpu).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    pcc = pearson_correlation_coefficient(out_ref_f, out_tt_cpu)

    logger.info("Output comparison:")
    logger.info(f"  Max diff: {out_max_diff:.6f}")
    logger.info(f"  Mean diff: {out_mean_diff:.6f}")
    logger.info(f"  PCC: {pcc:.6f}")

    state_diff = (state_ref - state_tt_cpu).abs()
    state_max_diff = state_diff.max().item()
    state_pcc = pearson_correlation_coefficient(state_ref, state_tt_cpu)

    logger.info("State comparison:")
    logger.info(f"  Max diff: {state_max_diff:.6f}")
    logger.info(f"  PCC: {state_pcc:.6f}")

    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(f"PASSED: ttnn matches HF reference (output PCC={pcc:.4f}, state PCC={state_pcc:.4f})")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("num_heads", [10, 32, 384])
def test_chunked_gdn_kernel_correctness(mesh_device, reset_seeds, num_heads):
    """Match HF recurrent GDN over ``SEQ_LEN`` (64) tokens vs naive ttnn stepping."""
    device = mesh_device
    batch_size = 2
    Dk, Dv = 128, 128
    num_cores = min(10, batch_size * num_heads)

    logger.info(f"Testing GDN: batch={batch_size}, num_heads={num_heads}, seq_len={SEQ_LEN}, Dk={Dk}, Dv={Dv}")

    torch.manual_seed(432)
    query = torch.randn(batch_size, num_heads, SEQ_LEN, Dk, dtype=torch.float32)
    key = torch.randn(batch_size, num_heads, SEQ_LEN, Dk, dtype=torch.float32)
    value = torch.randn(batch_size, num_heads, SEQ_LEN, Dv, dtype=torch.float32)
    g = torch.randn(batch_size, num_heads, SEQ_LEN, dtype=torch.float32) * 0.5 - 1.0
    beta = torch.randn(batch_size, num_heads, SEQ_LEN, dtype=torch.float32)

    out_ref, state_ref = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # out_ref: [B, H, S, Dv] (same layout as ``core_attn_out`` in the reference loop)
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    q_f, k_f, v_f, beta_f, g_f = preprocess_like_hf(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)
    # Shapes [B, H, S, *]
    num_pairs = batch_size * num_heads

    def to_tt(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    state_tt = ttnn.from_torch(
        torch.zeros(batch_size, num_heads, Dk, Dv, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output_tt = ttnn.from_torch(
        torch.zeros(batch_size, num_heads, 1, Dv, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    q_tt = to_tt(q_f)
    k_tt = to_tt(k_f)
    v_tt = to_tt(v_f)
    g_tt = to_tt(g_f)
    beta_tt = to_tt(beta_f)

    # step_outputs: list[torch.Tensor] = []
    # for t in range(SEQ_LEN):
    #     q_t =       q_tt[:, :, t:t+1]
    #     k_t =       k_tt[:, :, t:t+1]
    #     v_t =       v_tt[:, :, t:t+1]
    #     g_t =       g_tt[:, :, t:t+1]
    #     beta_t = beta_tt[:, :, t:t+1]

    #     output_tt = gdn_recurrence_fused_inplace(
    #         q_t,
    #         k_t,
    #         v_t,
    #         g_t,
    #         beta_t,
    #         state_tt,
    #         num_cores=num_cores,
    #         iter=t,
    #     )
    #     step_outputs.append(ttnn.to_torch(output_tt).float().squeeze(1).clone())

    out_tt_stacked = chunked_gdn_recurrence_fused_inplace(q_tt, k_tt, v_tt, g_tt, beta_tt, state_tt)
    out_tt_cpu = out_tt_stacked.reshape(batch_size, num_heads, SEQ_LEN, Dv).contiguous()
    state_tt_cpu = ttnn.to_torch(state_tt).float()

    logger.info(f"ttnn output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    out_ref_f = out_ref.float()
    out_diff = (out_ref_f - out_tt_cpu).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    pcc = pearson_correlation_coefficient(out_ref_f, out_tt_cpu)

    logger.info("Output comparison:")
    logger.info(f"  Max diff: {out_max_diff:.6f}")
    logger.info(f"  Mean diff: {out_mean_diff:.6f}")
    logger.info(f"  PCC: {pcc:.6f}")

    state_diff = (state_ref - state_tt_cpu).abs()
    state_max_diff = state_diff.max().item()
    state_pcc = pearson_correlation_coefficient(state_ref, state_tt_cpu)

    logger.info("State comparison:")
    logger.info(f"  Max diff: {state_max_diff:.6f}")
    logger.info(f"  PCC: {state_pcc:.6f}")

    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(f"PASSED: ttnn matches HF reference (output PCC={pcc:.4f}, state PCC={state_pcc:.4f})")
