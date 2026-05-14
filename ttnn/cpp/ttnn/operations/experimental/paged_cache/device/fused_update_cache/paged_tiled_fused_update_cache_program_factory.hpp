// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "paged_fused_update_cache_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

struct PagedTiledFusedUpdateCacheProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value);
};

struct PagedTiledFusedUpdateCacheMeshWorkloadFactory {
    // Per-coord program build.  See PagedRowMajorFusedUpdateCacheMeshWorkloadFactory.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PagedFusedUpdateCacheParams& operation_attributes,
        const PagedFusedUpdateCacheInputs& tensor_args,
        PagedFusedUpdateCacheResult& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::experimental::prim
