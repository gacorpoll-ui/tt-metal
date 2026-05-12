# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Tests for ttnn.sum with INT32 and float dtypes.
#
# INT32 background (issue #26724, plan in #43736):
#   The FPU GMPOOL primitive that ttnn's reduce ops historically relied on
#   silently produces zeros for INT32 inputs.  The SUM extension of the SFPU
#   reduce path (compute_kernel_lib::reduce_sfpu, compiled into
#   reduce_sfpu.cpp) handles INT32 SUM along H or W; full-tensor (HW)
#   reductions are decomposed at the prim layer into a W reduce followed by an
#   H reduce (see reduce_op.cpp's use_two_step_hw_reduce_for_int32 flag).
#
# Float coverage:
#   bfloat16 is the canonical on-device float dtype.  float32 exercises the
#   full-precision path.  Both use assert_with_pcc / assert_allclose rather
#   than exact equality because floating-point reduction order is not
#   guaranteed to match PyTorch's.

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

pytestmark = pytest.mark.use_module_device


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Shapes exercised by both INT32 and float parametrised tests.
# Covers: single tile, partial tiles, multi-batch, issue-#26724 repro shape.
_SHAPES = [
    (1, 1, 32, 32),  # single tile — issue #26724-style smallest case
    (1, 1, 64, 60),  # 2×2 tiles, partial W tile
    (1, 1, 100, 120),  # 4×4 tiles, partial H and W tiles
    (1, 1, 30, 96),  # Ht=1, Wt=3  (no H accumulation)
    (1, 1, 90, 32),  # Ht=3, Wt=1  (no W accumulation)
    (2, 3, 64, 64),  # multi-batch, whole tiles
    (1, 3, 17, 19),  # exact #26724 repro shape (non-tile-aligned, NC=3)
    (2, 4, 64, 60),  # multi-batch + partial W tile
]

# Reduction dimensions exercised by both parametrised tests.
# -1/-2 → W/H single-axis SFPU path
# 0/1   → batch/channel axes (transpose-then-reduce slow path)
# (-1,-2)/None → full-tensor W-then-H decomposition
_DIMS = [
    -1,  # W axis  → MULTI_CORE_W
    -2,  # H axis  → MULTI_CORE_H
    0,  # batch axis (slow path for INT32; fast path for float)
    1,  # channel axis (slow path for INT32; fast path for float)
    (-1, -2),  # HW axes → two-step decomposition
    None,  # all axes → two-step decomposition
]


# ---------------------------------------------------------------------------
# INT32 tests
# ---------------------------------------------------------------------------

# Small absolute bound so that the largest tensor (2×4×64×60 = 30 720 elems)
# cannot overflow int32: |sum| < 30 720 × 1 000 ≈ 3 × 10^7 << 2^31.
_INT32_VALUE_BOUND = 1000


@pytest.mark.parametrize("input_shape", _SHAPES)
@pytest.mark.parametrize("dim", _DIMS)
def test_sum_int32(device, input_shape, dim):
    """ttnn.sum on INT32 tensors must match torch.sum exactly."""
    torch.manual_seed(0)

    torch_input = torch.randint(
        -_INT32_VALUE_BOUND,
        _INT32_VALUE_BOUND + 1,
        input_shape,
        dtype=torch.int32,
    )
    # torch.sum on int32 promotes to int64; keep both sides in int64.
    torch_output = torch.sum(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    ttnn_output = ttnn.to_torch(ttnn.sum(ttnn_input, dim=dim))

    assert ttnn_output.dtype == torch.int32, f"Expected int32 output, got {ttnn_output.dtype}"

    # Align dtype (torch returns int64) and shape (ttnn keeps trailing
    # singleton dims for tile alignment) before exact comparison.
    actual = ttnn_output.to(torch.int64).reshape(torch_output.shape)
    assert_equal(actual, torch_output)


def test_sum_int32_all_zeros_regression(device):
    """Regression test for the silent all-zeros result on INT32 sum (issue #26724).

    Kept as a standalone test so a future regression on this exact failure
    mode is immediately visible in CI without needing to scan parametrised IDs.
    """
    torch.manual_seed(0)
    x = torch.randint(-_INT32_VALUE_BOUND, _INT32_VALUE_BOUND + 1, (1, 3, 17, 19), dtype=torch.int32)
    x_ttnn = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)

    # (A) Reduce over all dims → scalar.
    torch_sum_all = torch.sum(x)
    ttnn_sum_all = ttnn.to_torch(ttnn.sum(x_ttnn))
    assert ttnn_sum_all.dtype == torch.int32
    assert_equal(ttnn_sum_all.to(torch.int64).reshape(torch_sum_all.shape), torch_sum_all)

    # (B) Reduce over a specific dim.
    torch_sum_dim = torch.sum(x, dim=2, keepdim=False)
    ttnn_sum_dim = ttnn.to_torch(ttnn.sum(x_ttnn, dim=2, keepdim=False))
    assert ttnn_sum_dim.dtype == torch.int32
    assert_equal(ttnn_sum_dim.to(torch.int64).reshape(torch_sum_dim.shape), torch_sum_dim)


# ---------------------------------------------------------------------------
# Float tests
# ---------------------------------------------------------------------------

# Tolerance for PCC comparison.  bfloat16 has ~2–3 decimal digits of
# precision, so we use a slightly looser threshold than float32.
_BFLOAT16_PCC = 0.999
_FLOAT32_PCC = 0.9999

# Absolute / relative tolerances for torch.testing.assert_close.
_BFLOAT16_ATOL = 1e-1
_BFLOAT16_RTOL = 1e-1
_FLOAT32_ATOL = 1e-4
_FLOAT32_RTOL = 1e-4


def _ttnn_dtype(torch_dtype: torch.dtype) -> ttnn.DataType:
    return {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32}[torch_dtype]


@pytest.mark.parametrize("input_shape", _SHAPES)
@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize(
    "torch_dtype, pcc_threshold, atol, rtol",
    [
        (torch.bfloat16, _BFLOAT16_PCC, _BFLOAT16_ATOL, _BFLOAT16_RTOL),
        (torch.float32, _FLOAT32_PCC, _FLOAT32_ATOL, _FLOAT32_RTOL),
    ],
    ids=["bfloat16", "float32"],
)
def test_sum_float(device, input_shape, dim, torch_dtype, pcc_threshold, atol, rtol):
    """ttnn.sum on float tensors must be numerically close to torch.sum."""
    torch.manual_seed(0)

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    torch_output = torch.sum(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=_ttnn_dtype(torch_dtype),
    )
    ttnn_output = ttnn.to_torch(ttnn.sum(ttnn_input, dim=dim))

    assert ttnn_output.dtype == torch_dtype, f"Expected {torch_dtype} output, got {ttnn_output.dtype}"

    actual = ttnn_output.reshape(torch_output.shape)

    # Primary gate: Pearson Correlation Coefficient.
    assert_with_pcc(actual, torch_output, pcc_threshold)

    # Secondary gate: element-wise tolerance check.
    torch.testing.assert_close(actual, torch_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "torch_dtype, pcc_threshold, atol, rtol",
    [
        (torch.bfloat16, _BFLOAT16_PCC, _BFLOAT16_ATOL, _BFLOAT16_RTOL),
        (torch.float32, _FLOAT32_PCC, _FLOAT32_ATOL, _FLOAT32_RTOL),
    ],
    ids=["bfloat16", "float32"],
)
def test_sum_float_keepdim(device, torch_dtype, pcc_threshold, atol, rtol):
    """keepdim=True must preserve the reduced axis as a size-1 dimension."""
    torch.manual_seed(0)
    shape = (2, 3, 64, 64)

    torch_input = torch.randn(shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=_ttnn_dtype(torch_dtype),
    )

    for dim in (-1, -2, 1, 0):
        torch_output = torch.sum(torch_input, dim=dim, keepdim=True)
        ttnn_output = ttnn.to_torch(ttnn.sum(ttnn_input, dim=dim, keepdim=True))

        assert ttnn_output.shape == torch_output.shape, (
            f"keepdim shape mismatch for dim={dim}: " f"expected {torch_output.shape}, got {ttnn_output.shape}"
        )
        assert_with_pcc(ttnn_output, torch_output, pcc_threshold)
        torch.testing.assert_close(ttnn_output, torch_output, atol=atol, rtol=rtol)
