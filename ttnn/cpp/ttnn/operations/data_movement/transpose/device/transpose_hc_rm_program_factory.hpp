// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct TransposeHCRMProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
