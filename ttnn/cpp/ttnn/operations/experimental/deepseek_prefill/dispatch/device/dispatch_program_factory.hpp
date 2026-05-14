// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include "dispatch_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

struct DispatchProgramFactory {
    using tensor_return_value_t = std::array<Tensor, 2>;

    // Workload-level resources allocated once per cache miss in prepare_resources()
    // and re-passed to every per-coord create_descriptor() call.  All three
    // GlobalSemaphores are referenced as absolute addresses by writer/reader
    // runtime args, so their device-side allocations must outlive the per-coord
    // program builds.
    //
    // GlobalSemaphore has no default constructor — wrap each in std::optional<>
    // to satisfy the framework's `resource_t{}` value-init in
    // DescriptorMeshWorkloadFactoryAdapter.  prepare_resources() always
    // populates all three before create_descriptor() reads them.
    struct Resources {
        std::optional<GlobalSemaphore> init_semaphore;
        std::optional<GlobalSemaphore> exit_semaphore;
        std::optional<GlobalSemaphore> cross_device_semaphore;
    };

    // Allocates the three GlobalSemaphores and runs the cross-device Synchronize
    // barrier.  Invoked ONCE per workload by DescriptorMeshWorkloadFactoryAdapter.
    static Resources prepare_resources(
        const DispatchParams& operation_attributes,
        const DispatchInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Per-coord program build.  workload_resources is the value returned from
    // prepare_resources(); mesh_dispatch_coordinate identifies which device in
    // the mesh this program targets (used to derive fabric node id, neighbor
    // lookups, and linearized mesh coord baked into compile-time args).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DispatchParams& operation_attributes,
        const DispatchInputs& tensor_args,
        tensor_return_value_t& tensor_return_value,
        Resources& workload_resources,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
