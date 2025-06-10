/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
//#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
//#include "xla/service/gpu/kernels/ck_gemm.h"
#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"
//#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
//#include "xla/tsl/platform/errors.h"
//#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"


namespace xla::gpu {

class CkGemmFusion : public CustomKernelFusion {
 public:
  absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
      const se::DeviceDescription& device,
      const HloComputation* computation) const final {
    auto* dot = DynCast<HloDotInstruction>(computation->root_instruction());
    if (dot == nullptr) {
      return absl::InternalError(
          "ck_gemm requires ROOT operation to be a dot");
    }

    PrimitiveType dot_type = dot->shape().element_type();

    auto* lhs = Cast<HloParameterInstruction>(dot->operand(0));
    auto* rhs = Cast<HloParameterInstruction>(dot->operand(1));

    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    PrimitiveType lhs_type = lhs->shape().element_type();
    PrimitiveType rhs_type = rhs->shape().element_type();

        return ::xla::gpu::kernel::gemm_universal::GetCkGemmKernels("ck_gemm", dot_type, lhs_type, rhs_type,
                            m, n, k, device);
  }
};

}  // namespace xla::gpu

XLA_REGISTER_CUSTOM_FUSION("ck_gemm",
                           ::xla::gpu::CkGemmFusion);
