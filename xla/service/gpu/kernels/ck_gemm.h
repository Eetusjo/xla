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

#ifndef XLA_SERVICE_GPU_KERNELS_CK_GEMM_H_
#define XLA_SERVICE_GPU_KERNELS_CK_GEMM_H_

//===-------------------------------------------------------------------------//
//                 ! ! ! ! !      WARNING      ! ! ! ! !                      //
//===-------------------------------------------------------------------------//
//                                                                            //
//   Do not add external dependencies to this header. Use only std library.   //
//                                                                            //
//===-------------------------------------------------------------------------//
//                 ! ! ! ! !      WARNING      ! ! ! ! !                      //
//===-------------------------------------------------------------------------//

#include <cstdint>
#include <optional>

namespace xla::gpu::kernel::gemm_universal {

//===----------------------------------------------------------------------===//
// Tag based GEMM dispatching
//===----------------------------------------------------------------------===//

// We use tag-based template specializations to carefully avoid including
// ck_tile headers into regular libraries, and specialize templates in separate
// ROCM build targets that have no dependencies on other parts of XLA or ABSL to
// enable parallel compilation and minimize recompilations on code changes.
//
// Here we re-define some of the enums and types defined in ck_tile to
// break a dependency on them from XLA.

struct F16xF16ToF16 {};
struct BF16xBF16ToBF16 {};

// Matches GemmKernelArguments in ck_tile
struct Arguments {
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t stride_A;
    int32_t stride_B;
    int32_t stride_C;
    int32_t k_batch;
};

struct ArgsIndices {
  int64_t lhs;
  int64_t rhs;
  int64_t out;
};

//===----------------------------------------------------------------------===//
// ck_tile Host Side Adaptor
//===----------------------------------------------------------------------===//

template <typename Tag>
struct Traits;

struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// This is a type-erased adaptor that has all details required for launching
// a ck_tile kernel on a device. At run time device kernel parameters is really
// just a bag of bytes that driver sends to a kernel, so we rely on it to hide
// ck templates inside individual build targets and don't leak them into
// XLA, as they contain device code and can't be parsed by regular clang.
template <typename Tag>
class Adaptor {
 public:
  Dim3 BlockDim(int32_t m, int32_t n, int32_t k) const;
  Dim3 ThreadDim() const;

  int32_t SharedMemoryBytes() const;

  void Initialize(void* params, const Arguments& args, int32_t device_sms) const;
};

//===----------------------------------------------------------------------===//
// ck_tile Device Side Adaptor
//===----------------------------------------------------------------------===//

// We keep device side adaptor separate from host side adaptor so that we could
// easily split host and device code compilation if needed.

template <typename Tag>
class DeviceKernel {
 public:
  void* symbol() const;
};

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CK_GEMM_H_
