// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "all_to_all_async_generic_device_operation_types.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct AllToAllAsyncGenericProgram {
    // Workload-level resources allocated once per cache miss in prepare_resources()
    // and re-passed to every per-coord create_descriptor() call.  Storing the
    // GlobalSemaphores here keeps their device-side allocations alive for the
    // lifetime of the cached workload — both are referenced by writer runtime
    // args as absolute addresses.
    //
    // GlobalSemaphore has no default constructor (it owns a device allocation),
    // so the framework's `resource_t{}` value-init in DescriptorMeshWorkloadFactoryAdapter
    // would fail to compile with raw GlobalSemaphore members.  Wrap each in
    // std::optional<>; prepare_resources() always populates both before
    // create_descriptor() reads them.
    struct Resources {
        std::optional<tt::tt_metal::GlobalSemaphore> init_barrier_semaphore;
        std::optional<tt::tt_metal::GlobalSemaphore> final_barrier_semaphore;
    };

    // Allocates the two GlobalSemaphores and runs the cross-device Synchronize
    // barrier.  Invoked ONCE per workload (before any per-coord program build)
    // by the DescriptorMeshWorkloadFactoryAdapter.
    static Resources prepare_resources(
        const AllToAllAsyncGenericParams& operation_attributes,
        const AllToAllAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value);

    // Per-coord program build.  workload_resources is the value returned from
    // prepare_resources(); mesh_dispatch_coordinate identifies which device in
    // the mesh this program targets.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllToAllAsyncGenericParams& operation_attributes,
        const AllToAllAsyncGenericInputs& tensor_args,
        Tensor& tensor_return_value,
        Resources& workload_resources,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
