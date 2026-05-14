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
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <vector>

namespace ttnn::operations::ccl {

namespace detail {

std::pair<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis);

}  // namespace detail

struct AllToAllDispatchDeviceOperation {
    enum AllToAllTransferType {
        FullPacket,  // All pages are sent to the intermediate buffer and then written to the output buffer later
        PageByPage,  // Each page is sent directly to the output buffer to conserve L1 space via intermediates
    };
    struct operation_attributes_t {
        const CoreRangeSet worker_core_range_set;
        const MemoryConfig output_mem_config;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        const AllToAllTransferType impl;
        const uint32_t output_concat_dim;
        static constexpr auto attribute_names = std::forward_as_tuple(
            "worker_core_range_set", "output_mem_config", "axis", "num_links", "topology", "impl", "output_concat_dim");
        auto attribute_values() const {
            return std::forward_as_tuple(
                worker_core_range_set, output_mem_config, axis, num_links, topology, impl, output_concat_dim);
        };
    };
    struct tensor_args_t {
        const Tensor input_tensor;
        const Tensor expert_indices_tensor;
        const Tensor expert_mapping_tensor;
        const std::optional<std::array<Tensor, 2>> optional_output_tensors;
    };

    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;

    using tensor_return_value_t = std::array<Tensor, 2>;

    struct AllToAllDispatchSparse {
        // Workload-level resources allocated once per cache miss in prepare_resources() and
        // re-passed to every per-coord create_descriptor() call.  Optional<> wrappers are
        // required because GlobalSemaphore has no default constructor but the framework
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

    using program_factory_t = std::variant<AllToAllDispatchSparse>;

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
ttnn::operations::ccl::AllToAllDispatchDeviceOperation::tensor_return_value_t all_to_all_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set,
    ttnn::operations::ccl::AllToAllDispatchDeviceOperation::AllToAllTransferType impl,
    uint32_t output_concat_dim);
}  // namespace ttnn::prim
