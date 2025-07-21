/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/ck_gemm_fusion.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

using stream_executor::CudaComputeCapability;

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Pattern matching tests
//===----------------------------------------------------------------------===//

// TODO

//===----------------------------------------------------------------------===//
// Run And Compare Tests
//===----------------------------------------------------------------------===//

TEST_F(CkFusionTest, RowMajorGemmKernel) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text_ref = R"(
  HloModule test

  ENTRY %main (p0: f16[100,784], p1: f16[7840,10]) -> f16[100,10] {
    %p0 = f16[100,784]{1,0} parameter(0)
    %p1 = f16[784,10]{1,0} parameter(1)
    ROOT %r = f16[100,10]{1,0} dot(%p0, %p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule ck

  ck_gemm {
    arg0 = f16[100,784]{1,0} parameter(0)
    arg1 = f16[784,10]{1,0} parameter(1)
    ROOT dot = f16[100,10]{1,0} dot(arg0, arg1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    arg0 = f16[100,784]{1,0} parameter(0)
    arg1 = f16[784,10]{1,0} parameter(1)
    ROOT _ = f16[100,10]{1,0} fusion(arg0, arg1), kind=kCustom, calls=ck_gemm,
      backend_config={"fusion_backend_config":{kind: "__custom_fusion", custom_fusion_config: {"name":"ck_gemm", "kernel_index":0}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_ref, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}


}  // namespace xla::gpu
