// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct NeighborPadAsyncMeshWorkloadFactory {
    // Workload-level resources.  The three GlobalSemaphores live on
    // NeighborPadAsyncParams (allocated by the Python-level caller) so we do
    // NOT need to allocate any resources here.  prepare_resources() is still
    // used so the cross-device Synchronize() barrier — required to flush
    // fabric writes from prior ops before any neighbor_pad program is built —
    // runs exactly once per workload (before any per-coord create_descriptor()
    // call), matching the legacy behaviour where Synchronize() lived at the
    // top of create_mesh_workload().
    struct Resources {};

    // Runs the cross-device Synchronize() barrier exactly once per workload.
    // Invoked by DescriptorMeshWorkloadFactoryAdapter before any per-coord
    // program build.
    static Resources prepare_resources(
        const NeighborPadAsyncParams& operation_attributes,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value);

    // Per-coord program build.  workload_resources is unused but present so
    // the adapter dispatches to the prepare_resources + create_descriptor
    // overload; mesh_dispatch_coordinate identifies which device in the mesh
    // this program targets (used to derive forward/backward neighbor coords
    // for the H- and W-fabric exchanges).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const NeighborPadAsyncParams& operation_attributes,
        const NeighborPadAsyncInputs& tensor_args,
        Tensor& tensor_return_value,
        Resources& workload_resources,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
