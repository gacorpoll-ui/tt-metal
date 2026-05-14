// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct SliceReshardAsyncProgramFactory {
    // Per-coord program build.  Both GlobalSemaphores live on SliceReshardAsyncParams
    // (allocated by the caller) so this factory needs no prepare_resources hook —
    // the semaphores are passed through and their addresses are written into runtime
    // args every dispatch via the normal apply_descriptor_runtime_args path.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SliceReshardAsyncParams& args,
        const Tensor& tensor_args,
        Tensor& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
