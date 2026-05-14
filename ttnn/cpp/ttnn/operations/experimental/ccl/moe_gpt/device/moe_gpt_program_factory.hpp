// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_gpt_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"

#include <optional>

namespace ttnn::operations::experimental::moe_gpt::program {

struct MoEGPTMeshWorkloadFactory {
    // Per-coord program build.  All buffer-address runtime args are registered
    // via KernelDescriptor::emplace_runtime_args(Buffer*), so the descriptor
    // framework patches them automatically on every dispatch — no
    // override_runtime_arguments hook is required.  The combine output CB
    // carries `.buffer` so the framework also patches its dynamic L1 address.
    // No GlobalSemaphores are used here (only local CreateSemaphore, translated
    // to SemaphoreDescriptor), so no prepare_resources hook is required either.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::experimental::moe_gpt::program
