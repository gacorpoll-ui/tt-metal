# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side modulated deformable conv 2D — starting scaffold using ttnn.grid_sample.

UniAD's ResNet101 backbone has 26 modulated deformable conv calls per
inference (stages 3 & 4 with `stage_with_dcn=(False, False, True, True)`).
The default path in TtModulatedDeformConv2dPack pulls x / offset / mask
to host and calls mmcv's `ext_module.modulated_deform_conv_forward` on
CPU — TT_DCN_TIMING=1 shows the CPU compute dominates that path.

This module decomposes modulated deformable conv (Dai et al. 2018) into
K*K (= 9 for K=3) `ttnn.grid_sample` calls + per-kernel-position
matmuls, so the whole op can stay on device and the surrounding ResNet
can eventually be captured by Metal Trace.

Math (per output (h_o, w_o), output channel c_out):

    acc = 0
    for kh in range(K):
      for kw in range(K):
        kk = kh*K + kw
        sy = h_o*stride[0] - padding[0] + kh*dilation[0] + offset_y[b, kk, h_o, w_o]
        sx = w_o*stride[1] - padding[1] + kw*dilation[1] + offset_x[b, kk, h_o, w_o]
        for c_in in range(C_in):
          sampled = bilinear_interp(x[b, c_in, :, :], sy, sx) * mask[b, kk, h_o, w_o]
          acc += weight[c_out, c_in, kh, kw] * sampled
    output[b, c_out, h_o, w_o] = acc + bias[c_out]

`ttnn.grid_sample` expects:
  - input NHWC, shape (N, H_in, W_in, C)
  - grid (N, H_out, W_out, 2) in (x, y) order normalized to [-1, 1]
  - mode="bilinear", align_corners=True (matches mmcv's behavior)

Known limitations (see tests/unit/test_dcn_device.py):
  - ttnn.grid_sample input channel width is bounded by TILE_WIDTH * 8 = 256;
    UniAD DCNs hit C_in = 1024 (stage 3) and 2048 (stage 4), which the
    current ttnn kernel rejects. Channel-chunking or a wider kernel is
    needed before this can replace the host path end-to-end.
  - bfloat16 bilinear + bfloat16 matmul accumulate enough error that
    the e2e PCC sdc_traj gate (0.99) is not yet met by this decomposition
    alone — fused or higher-precision accumulators are needed.
"""

import torch
import ttnn


class TtModulatedDeformConv2dDevice:
    """Device-side modulated deformable conv 2D.

    Stateful so per-instance constants (base grids, normalized weight
    slices, bias) can be uploaded once and reused across warm calls.
    """

    def __init__(
        self,
        weight,  # torch.Tensor (C_out, C_in, K, K)
        bias,  # torch.Tensor (C_out,) or None
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
        device,
    ):
        assert groups == 1, "device DCN prototype only supports groups=1"
        assert deform_groups == 1, "device DCN prototype only supports deform_groups=1"

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.C_out, self.C_in, self.K, _ = weight.shape
        assert weight.shape[2] == weight.shape[3], "kernel must be square"

        # Per-kernel-position weight slices, each shape (C_out, C_in),
        # uploaded as TILE bfloat16 for fast matmul.
        self.weight_per_kp = []
        for kh in range(self.K):
            for kw in range(self.K):
                w_kk = weight[:, :, kh, kw].contiguous()
                # Pre-transpose so matmul input shape is (M, C_in) × (C_in, C_out).
                w_kk_t = w_kk.t().contiguous()  # (C_in, C_out)
                self.weight_per_kp.append(
                    ttnn.from_torch(w_kk_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                )

        if bias is not None:
            self.bias = ttnn.from_torch(bias.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        else:
            self.bias = None

        # Base grid is built lazily on first call (depends on H_out/W_out,
        # which we only learn at runtime).
        self._base_grid_cache = None  # dict keyed by (H_in, W_in, H_out, W_out)

    def _build_base_grid(self, H_in, W_in, H_out, W_out, batch):
        """Build the per-kernel-position normalized base grids.

        For each kp (kh, kw), the base sample position in input space is:
            sy_base = h_o * stride[0] - padding[0] + kh * dilation[0]
            sx_base = w_o * stride[1] - padding[1] + kw * dilation[1]

        Normalizing for `align_corners=True`:
            gy = sy_base / (H_in - 1) * 2 - 1
            gx = sx_base / (W_in - 1) * 2 - 1

        Returns a list of K*K tuples (gx_base, gy_base), each a torch
        tensor of shape (batch, H_out, W_out) ready for upload.
        """
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # Coordinates (h_o, w_o) for each output position
        h_o_coords = torch.arange(H_out, dtype=torch.float32)
        w_o_coords = torch.arange(W_out, dtype=torch.float32)
        h_grid, w_grid = torch.meshgrid(h_o_coords, w_o_coords, indexing="ij")
        # (H_out, W_out)

        base_grids = []
        for kh in range(self.K):
            for kw in range(self.K):
                sy_base = h_grid * sh - ph + kh * dh
                sx_base = w_grid * sw - pw + kw * dw

                gy_base = sy_base / max(H_in - 1, 1) * 2.0 - 1.0
                gx_base = sx_base / max(W_in - 1, 1) * 2.0 - 1.0

                gy_base_b = gy_base.unsqueeze(0).expand(batch, H_out, W_out).contiguous()
                gx_base_b = gx_base.unsqueeze(0).expand(batch, H_out, W_out).contiguous()
                base_grids.append((gx_base_b, gy_base_b))

        # Also compute the offset scale factors (so per-call offset can
        # be normalized cheaply with a single multiply).
        gy_scale = 2.0 / max(H_in - 1, 1)
        gx_scale = 2.0 / max(W_in - 1, 1)
        return base_grids, gy_scale, gx_scale

    def __call__(self, x_nhwc, offset_y_nhwc, offset_x_nhwc, mask_nhwc):
        """Forward.

        Args:
          x_nhwc: (B, H_in, W_in, C_in) bfloat16 NHWC on device
          offset_y_nhwc: (B, H_out, W_out, K*K) bfloat16 NHWC — y-offsets per kp
          offset_x_nhwc: (B, H_out, W_out, K*K) bfloat16 NHWC — x-offsets per kp
          mask_nhwc: (B, H_out, W_out, K*K) bfloat16 NHWC — modulation masks

        Returns:
          (B, H_out, W_out, C_out) bfloat16 NHWC on device
        """
        B, H_in, W_in, C_in = x_nhwc.shape
        _, H_out, W_out, _ = offset_y_nhwc.shape
        K = self.K

        cache_key = (B, H_in, W_in, H_out, W_out)
        if self._base_grid_cache is None or cache_key not in self._base_grid_cache:
            base_grids, gy_scale, gx_scale = self._build_base_grid(H_in, W_in, H_out, W_out, B)
            # Upload base grids once per (input-shape, output-shape) combo.
            base_grids_dev = []
            for gx_base_b, gy_base_b in base_grids:
                # ttnn.grid_sample wants grid as (B, H_out, W_out, 2) with (x, y) order.
                grid_xy = torch.stack([gx_base_b, gy_base_b], dim=-1)
                base_grids_dev.append(
                    ttnn.from_torch(grid_xy, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                )
            self._base_grid_cache = {cache_key: (base_grids_dev, gy_scale, gx_scale)}

        base_grids_dev, gy_scale, gx_scale = self._base_grid_cache[cache_key]

        # Per-kernel-position loop. Each iteration: build sample grid →
        # grid_sample → mask multiply → matmul → accumulate.
        output = None
        for kk in range(K * K):
            # Per-call grid additions: (B, H_out, W_out, 1) bfloat16
            oy = offset_y_nhwc[..., kk : kk + 1]
            ox = offset_x_nhwc[..., kk : kk + 1]
            m_kk = mask_nhwc[..., kk : kk + 1]

            # Scale offsets to normalized grid space.
            oy_norm = ttnn.multiply(oy, gy_scale)
            ox_norm = ttnn.multiply(ox, gx_scale)

            # Build (x, y) grid: base + scaled offset
            # base_grids_dev[kk] shape: (B, H_out, W_out, 2) with [x, y]
            gx_base_full = base_grids_dev[kk][..., 0:1]
            gy_base_full = base_grids_dev[kk][..., 1:2]
            gx = ttnn.add(gx_base_full, ox_norm)
            gy = ttnn.add(gy_base_full, oy_norm)
            grid = ttnn.concat([gx, gy], dim=-1)  # (B, H_out, W_out, 2)

            # grid_sample expects ROW_MAJOR_LAYOUT input + grid
            x_rm = ttnn.to_layout(x_nhwc, ttnn.ROW_MAJOR_LAYOUT)
            sampled = ttnn.grid_sample(x_rm, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            # sampled shape: (B, H_out, W_out, C_in)

            # Apply mask (broadcast across C_in)
            weighted = ttnn.multiply(sampled, m_kk)

            # Matmul along C_in dim: (B, H_out, W_out, C_in) × (C_in, C_out)
            weighted_tile = ttnn.to_layout(weighted, ttnn.TILE_LAYOUT)
            out_kk = ttnn.matmul(weighted_tile, self.weight_per_kp[kk])

            output = out_kk if output is None else ttnn.add(output, out_kk)

        if self.bias is not None:
            output = ttnn.add(output, self.bias)

        return output
