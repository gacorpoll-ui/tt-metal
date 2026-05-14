// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "strided_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <optional>

namespace ttnn::experimental::prim {

// Descriptor-based mesh-workload factory.
//
// - create_descriptor: Per-coord descriptor build invoked by the framework via
//   the DescriptorMeshWorkloadFactoryAdapter.  All buffer-address runtime args
//   are registered via emplace_runtime_args(Buffer*) so the framework patches
//   them on every dispatch -- no override_runtime_arguments hook is required.
//   GlobalSemaphores are owned by the caller and live on
//   StridedAllGatherAsyncParams, so no prepare_resources hook is required
//   either.  Matmul-fusion (fused_op_signaler) is NOT supported in this
//   code path.
//
// - shared_variables_t / strided_all_gather_async_minimal_default_helper /
//   override_runtime_arguments_per_program: legacy Program&-based helpers,
//   preserved verbatim as a parallel API for the matmul-fusion consumer
//   (experimental/ccl/strided_all_gather_minimal_matmul_async).  These are
//   NOT invoked from create_descriptor; the descriptor path is a
//   self-contained re-implementation.  Once the matmul-fusion variant is
//   migrated to ProgramDescriptor, the helper and shared_variables_t can be
//   removed.
struct StridedAllGatherAsyncProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
        std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
        std::vector<CoreCoord> all_cores;
        uint32_t num_links;
        uint32_t num_directions_per_link;
        uint32_t num_workers_per_direction;
        uint32_t num_mux_cores_per_direction_per_link;
        uint32_t num_cores_per_link;
    };

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const StridedAllGatherAsyncParams& operation_attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

    static shared_variables_t strided_all_gather_async_minimal_default_helper(
        tt::tt_metal::Program& program,
        const Tensor& input_tensor,
        const MeshCoordinate& sender_device_coord,
        const std::optional<MeshCoordinate>& forward_coord,
        const std::optional<MeshCoordinate>& backward_coord,
        Tensor& output_tensor,
        uint32_t dim,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index,
        ttnn::ccl::Topology topology,
        const std::vector<GlobalSemaphore>& semaphore,
        std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler>& fused_op_signaler,
        bool read_local_slice_from_input,
        std::optional<uint32_t> num_workers_per_direction_opt,
        std::optional<uint32_t> num_buffers_per_channel,
        std::optional<uint32_t> mm_cores_y,
        std::optional<uint32_t> mm_block_ht,
        std::optional<uint32_t> mm_block_wt,
        CoreCoord core_grid_offset = CoreCoord(0, 0));

    static void override_runtime_arguments_per_program(
        const shared_variables_t& shared_variables,
        tt::tt_metal::Program& program,
        const StridedAllGatherAsyncParams& attributes,
        const StridedAllGatherAsyncInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::experimental::prim
