// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rms_allgather_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct RMSAllGatherMeshWorkloadFactory {
    // Per-coord program build.  The GlobalSemaphore lives on RMSAllGatherParams
    // (allocated by the caller), so no prepare_resources hook is required --
    // the semaphore is passed through and its address is written into runtime
    // args every dispatch via the normal apply_descriptor_runtime_args path.
    // No Buffer* runtime args either, so the cached-workload cache-hit path
    // takes the descriptor-rebuild route, which re-reads
    // operation_attributes.semaphore.address() on every dispatch and patches
    // it in.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RMSAllGatherParams& operation_attributes,
        const RMSAllGatherInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
