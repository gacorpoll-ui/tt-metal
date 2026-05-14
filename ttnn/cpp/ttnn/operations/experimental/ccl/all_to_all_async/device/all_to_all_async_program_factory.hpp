// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct AllToAllAsyncProgramFactory {
    // Per-coord program build.  The GlobalSemaphore lives on AllToAllAsyncParams
    // (allocated by the caller) so this factory needs no prepare_resources hook —
    // the semaphore is passed through and its address is written into runtime
    // args every dispatch via the normal apply_descriptor_runtime_args path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllToAllAsyncParams& operation_attributes,
        const AllToAllAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
