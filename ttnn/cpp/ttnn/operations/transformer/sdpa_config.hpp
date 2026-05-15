// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::optional<CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    // Global Q scheduling: treat (batch, head, q_chunk) as one linear space and split it evenly
    // across cores instead of the hierarchical batch -> heads -> q_chunks split. Use for workloads
    // where the hierarchical split leaves cores idle (e.g. low batch x head product). Default
    // (false) keeps the hierarchical parallelization. Supports both causal and non-causal paths;
    // chunked prefill and attention_sink are not supported under global_q_scheduling. When causal
    // with an even q_num_chunks, a zigzag remap pairs light/heavy q_chunks per core for load
    // balancing.
    bool global_q_scheduling = false;
};

}  // namespace ttnn::operations::transformer
