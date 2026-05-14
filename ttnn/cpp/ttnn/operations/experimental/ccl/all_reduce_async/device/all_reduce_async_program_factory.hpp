// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_reduce_async_device_operation_types.hpp"

#include "ttnn/distributed/types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct AllReduceAsyncMeshWorkloadFactory {
    // Per-coord program build.  The GlobalSemaphore lives on
    // AllReduceAsyncParams (allocated by the caller), so no prepare_resources
    // hook is required -- the semaphore is passed through and its address is
    // written into runtime args every dispatch via the normal
    // apply_descriptor_runtime_args path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllReduceAsyncParams& operation_attributes,
        const AllReduceAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim

namespace ttnn {

// Worker-core picker shared with llama_all_gather_matmul_async.  Kept at this
// (legacy) header location so external consumers continue to compile after the
// descriptor migration of the program factory.
std::tuple<CoreRangeSet, std::vector<CoreCoord>> ar_choose_worker_cores(
    size_t num_links, size_t num_workers_per_link, const CoreRangeSet& available_cores);

}  // namespace ttnn
