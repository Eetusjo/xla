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

//#include "xla/service/gpu/kernels/ck_gemm_fusion.h"

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class CkFusionTest : public HloTestBase {
 public:
//  int GpuSharedMemorySize() {
//    return backend()
//        .default_stream_executor()
//        ->GetDeviceDescription()
//        .shared_memory_per_block_optin();
//  }
//  int CutlassGemmKernelSharedMemorySize(PrimitiveType dot_type,
//                                        PrimitiveType lhs_type,
//                                        PrimitiveType rhs_type, int m, int n,
//                                        int k) {
//    return kernel::gemm_universal::GetCutlassGemmKernels(
//               "cutlass_gemm", dot_type, lhs_type, rhs_type, m, n, k,
//               /*indices=*/{0, 1, 2}, /*slices=*/{},
//               backend().default_stream_executor()->GetDeviceDescription())
//        ->at(0)
//        .shared_memory_bytes();
//  };
};

//===----------------------------------------------------------------------===//
// Run And Compare Tests
//===----------------------------------------------------------------------===//

TEST_F(CkFusionTest, RowMajorGemmKernel) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text = R"(
    HloModule test

    ENTRY %main (p0: bf16[100,784], p1: bf16[784,10]) -> bf16[100,10] {
      %p0 = bf16[100,784]{1,0} parameter(0)
      %p1 = bf16[784,10]{1,0} parameter(1)
      ROOT %r = bf16[100,10]{1,0} dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* hlo_text_custom_fusion = R"(
  HloModule ck

  ck_gemm {
    arg0 = bf16[100,784]{1,0} parameter(0)
    arg1 = bf16[784,10]{1,0} parameter(1)
    ROOT dot = bf16[100,10]{1,0} dot(arg0, arg1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    arg0 = bf16[100,784]{1,0} parameter(0)
    arg1 = bf16[784,10]{1,0} parameter(1)
    ROOT _ = bf16[100,10]{1,0} fusion(arg0, arg1), kind=kCustom, calls=ck_gemm,
      backend_config={"fusion_backend_config":{kind: "__custom_fusion", custom_fusion_config: {"name":"ck_gemm", "kernel_index":0}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}


}  // namespace xla::gpu
