// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

struct AllToAllCombineDeviceOperation {
    struct operation_attributes_t {
        const MemoryConfig output_mem_config;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        const bool locally_reduced;
        const CoreRangeSet worker_core_range_set;
        const uint32_t output_shard_dim;
        static constexpr auto attribute_names = std::forward_as_tuple(
            "output_mem_config",
            "axis",
            "num_links",
            "topology",
            "locally_reduced",
            "worker_core_range_set",
            "output_shard_dim");
        auto attribute_values() const {
            return std::forward_as_tuple(
                output_mem_config, axis, num_links, topology, locally_reduced, worker_core_range_set, output_shard_dim);
        };
    };
    struct tensor_args_t {
        const ttnn::Tensor input_tensor;
        const ttnn::Tensor mapping_tensor;
        const ttnn::Tensor metadata_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = ttnn::Tensor;

    struct AllToAllCombineFromSparse {
        // Workload-level resources allocated once per cache miss in prepare_resources()
        // and re-passed to every per-coord create_descriptor() call.  Optional<> wrappers
        // are required because GlobalSemaphore has no default constructor but the framework
        // value-initializes resource_t in shared_variables_t.
        struct Resources {
            std::optional<tt::tt_metal::GlobalSemaphore> init_semaphore;
            std::optional<tt::tt_metal::GlobalSemaphore> cross_device_semaphore;
        };

        // Allocates the two GlobalSemaphores and runs the cross-device Synchronize barrier.
        // Invoked ONCE per workload before any per-coord program build.
        static Resources prepare_resources(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // Per-coord program build.  workload_resources comes from prepare_resources();
        // mesh_dispatch_coordinate identifies which device in the mesh this program targets.
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            Resources& workload_resources,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<AllToAllCombineFromSparse>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor all_to_all_combine(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    bool locally_reduced,
    const CoreRangeSet& worker_core_range_set,
    uint32_t output_shard_dim);
}  // namespace ttnn::prim
