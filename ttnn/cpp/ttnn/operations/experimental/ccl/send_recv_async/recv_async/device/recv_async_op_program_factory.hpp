// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recv_async_op_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>
#include <vector>

namespace ttnn::experimental::prim {

struct RecvAsyncMeshWorkloadFactory {
    // Per-coord program build.  The MeshSocket lives on RecvAsyncParams (caller
    // allocated) so this factory needs no prepare_resources hook — the socket
    // config buffer address is written into the kernel runtime args every
    // dispatch via the normal apply_descriptor_runtime_args path.
    //
    // For coordinates that are not receiver devices for the socket, an empty
    // ProgramDescriptor is returned: no kernels, no CBs.  The legacy
    // create_mesh_workload pre-filtered such coords out via
    // get_workload_coords<RECEIVER>; in the descriptor pattern the framework
    // iterates every tensor coord, so we emit a no-op program for non-receiver
    // coords — mirrors the send_async pattern.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RecvAsyncParams& operation_attributes,
        const Tensor& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
