// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device operation header for AllGatherConcat (heads-fused) using the
// ProgramDescriptor pattern.

#include "ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/device/all_gather_concat_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/distributed/types.hpp"

#include <optional>

namespace ttnn::experimental::prim {

struct AllGatherConcatDeviceOperation {
    using operation_attributes_t = AllGatherConcatParams;
    using tensor_args_t = AllGatherConcatInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    // Per-coord program build.  mesh_dispatch_coordinate identifies which device
    // in the mesh this program targets; the framework iterates over all coords
    // in the workload and calls create_descriptor() once per coord on a cache
    // miss.  Buffer addresses in the descriptor are patched automatically on
    // cache hits via the BufferBindings recorded in emplace_runtime_args().
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    // GlobalSemaphore in operation_attributes is not hashable via the default
    // reflection path; hash the rest of the fields explicitly + tensor_args.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor all_gather_concat(
    const Tensor& input_tensor,
    Tensor& buffer_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const GlobalSemaphore& global_semaphore,
    uint32_t num_heads,
    const MemoryConfig& memory_config,
    bool use_noc1_only,
    std::optional<uint32_t> num_links,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id);

}  // namespace ttnn::experimental::prim
