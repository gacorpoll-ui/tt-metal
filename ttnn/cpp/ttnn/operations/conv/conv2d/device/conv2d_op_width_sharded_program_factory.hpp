// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/storage.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// ProgramDescriptor-based factory for the width-sharded Conv2d variant.
//
// Lives in the multi-variant `Conv2dDeviceOperation::program_factory_t` next to the legacy
// Conv2dShardedProgramFactory. The framework's adapter dispatches to the descriptor or legacy
// path per alternative, so this factory only needs to satisfy ProgramDescriptorFactoryConcept
// (i.e. expose `create_descriptor` and no `create` / `override_runtime_arguments`).
//
// The sliding-window indices config tensor must outlive create_descriptor() (it backs either a
// globally-allocated L1 CB or a DRAM-resident reader buffer that the kernel addresses by
// compile-time constant). The framework's optional `prepare_resources` hook is used to allocate
// it once on program-cache miss; the returned DeviceStorage is kept alive in the cached mesh
// workload and handed back to create_descriptor on subsequent calls.
struct Conv2dWidthShardedProgramFactory {
    // Allocates the sliding-window reader-indices config tensor on the device and returns its
    // DeviceStorage. The framework stores this in the cached mesh workload and re-uses it for
    // every cache hit, guaranteeing the indices buffer stays alive for the program's lifetime.
    static tt::tt_metal::DeviceStorage prepare_resources(
        const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor);

    // Builds the ProgramDescriptor for the width-sharded conv2d program. `resources` holds the
    // DeviceStorage produced by prepare_resources(); its buffer is used both as the backing
    // buffer for the (optional) L1-resident READER_INDICES CB and to embed the DRAM-resident
    // config buffer address into the activation kernel's compile-time args.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const Conv2dParams& operation_attributes,
        const Conv2dInputs& tensor_args,
        Tensor& output_tensor,
        tt::tt_metal::DeviceStorage& resources);
};

}  // namespace ttnn::prim
