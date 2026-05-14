// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once

#include "ttnn/operations/ccl/ccl_common.hpp"

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/types.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>

namespace ttnn {
namespace operations::point_to_point {

struct PointToPointOp {
    struct operation_attributes_t {
        const MeshCoordinate& receive_coord;
        const MeshCoordinate& send_coord;
        const ::ttnn::ccl::Topology topology;

        // put this in here to hash on tensor spec
        const ttnn::TensorSpec _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("send_coord", "receive_coord", "topology");
        auto attribute_values() const { return std::forward_as_tuple(send_coord, receive_coord, topology); };
    };

    struct tensor_args_t {
        const Tensor input_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
        const std::optional<ttnn::Tensor> optional_intermediate_tensor;
    };

    // entry 0 is the intermediate. Entry 1 is the final output
    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;

    struct SendReceive {
        // Workload-level resources allocated once per cache miss in prepare_resources()
        // and re-passed to every per-coord create_descriptor() call.  The single
        // GlobalSemaphore is shared between the send and receive programs on the
        // two endpoint devices — both reference its absolute address in runtime
        // args.  GlobalSemaphore has no default constructor (it owns a device
        // allocation), so the framework's `resource_t{}` value-init in
        // DescriptorMeshWorkloadFactoryAdapter would fail with a raw member.
        // Wrap in std::optional<>; prepare_resources() always populates it
        // before create_descriptor() reads it.
        struct Resources {
            std::optional<tt::tt_metal::GlobalSemaphore> semaphore;
        };

        // Allocates the GlobalSemaphore and runs the cross-device Synchronize
        // barrier.  Invoked ONCE per workload (before any per-coord program
        // build) by the DescriptorMeshWorkloadFactoryAdapter.
        static Resources prepare_resources(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // Per-coord program build.  workload_resources is the value returned
        // from prepare_resources(); mesh_dispatch_coordinate identifies which
        // device in the mesh this program targets (used to dispatch to either
        // the send or receive sub-program factory).
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            Resources& workload_resources,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    using program_factory_t = std::variant<SendReceive>;

    // Mandatory methods

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
};

namespace detail {

struct AlignedPacketDims {
    const uint32_t packet_size_bytes;
    const uint32_t max_num_pages_per_packet;
    const uint32_t num_page_segments;
    const uint32_t total_packets;
};

AlignedPacketDims compute_aligned_packet_dims(
    const DataType& dtype, uint32_t page_size_bytes, uint32_t num_pages, uint32_t alignment);

struct Fabric1DRoute {
    const uint32_t num_hops;
    const bool is_forward;
    const tt::tt_fabric::FabricNodeId neighbor_id;
};

Fabric1DRoute fabric_1d_routing(
    const MeshDevice* mesh_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& receiver_coord,
    ::ttnn::ccl::Topology topology);

}  // namespace detail

tt::tt_metal::ProgramDescriptor send_program_factory(
    const PointToPointOp::tensor_args_t& tensor_args,
    const PointToPointOp::operation_attributes_t& operation_attributes,
    const MeshCoordinate& send_coord,
    const MeshCoordinate& receive_coord,
    PointToPointOp::tensor_return_value_t& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore);

tt::tt_metal::ProgramDescriptor receive_program_factory(
    const PointToPointOp::operation_attributes_t& operation_attributes,
    PointToPointOp::tensor_return_value_t& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore);
}  // namespace operations::point_to_point

namespace prim {
ttnn::operations::point_to_point::PointToPointOp::tensor_return_value_t point_to_point(
    const Tensor& input_tensor,
    const ::ttnn::ccl::Topology& topology,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor = std::nullopt);
}  // namespace prim
}  // namespace ttnn
