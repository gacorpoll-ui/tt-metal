// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "combine_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include <ttnn/global_semaphore.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine {

struct CombineProgramFactory {
    // Workload-level resources allocated once per cache miss in prepare_resources()
    // and re-passed to every per-coord create_descriptor() call.  Storing the two
    // GlobalSemaphores here keeps their device-side allocations alive for the
    // lifetime of the cached workload — both are referenced by writer runtime
    // args as absolute addresses.
    //
    // GlobalSemaphore has no default constructor, so wrap in std::optional<> to
    // satisfy the framework's `resource_t{}` value-init in DescriptorMeshWorkloadFactoryAdapter.
    // prepare_resources() always populates both before create_descriptor() reads them.
    struct Resources {
        std::optional<GlobalSemaphore> init_semaphore;
        std::optional<GlobalSemaphore> exit_semaphore;
    };

    // Allocates the two GlobalSemaphores and runs the cross-device Synchronize
    // barrier.  Invoked ONCE per workload (before any per-coord program build)
    // by the DescriptorMeshWorkloadFactoryAdapter.
    static Resources prepare_resources(
        const CombineParams& operation_attributes, const CombineInputs& tensor_args, ttnn::Tensor& tensor_return_value);

    // Per-coord program build.  workload_resources is the value returned from
    // prepare_resources(); mesh_dispatch_coordinate identifies which device in
    // the mesh this program targets (used to derive fabric node id, neighbor
    // lookups, and counter_offset).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const CombineParams& operation_attributes,
        const CombineInputs& tensor_args,
        ttnn::Tensor& tensor_return_value,
        Resources& workload_resources,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
