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

#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"
//
//#include <cstddef>
//#include <cstdint>
//#include <memory>
//#include <optional>
//#include <string>
//#include <tuple>
//#include <utility>
//#include <vector>
//
#include "absl/container/flat_hash_map.h"
//#include "absl/log/log.h"
//#include "absl/status/status.h"
//#include "absl/strings/str_cat.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/ck_gemm.h"
//#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
//#include "xla/stream_executor/launch_dim.h"
//#include "xla/xla_data.pb.h"

namespace xla::gpu::kernel::gemm_universal {

//// Each individual CUTLASS kernel adaptor will be compiled in a separate
//// cuda_library and linked into the `cutlass_gemm_custom_kernels` target. We use
//// this approach for a few reasons:
////
////   - It enables parallel compilation of CUTLASS templates which in practice
////     becomes quite expensive for any non-trivial GEMM.
////
////   - We do not include any of the CUTLASS headers in our custom kernel
////     library which would require converting it to a cuda_library, and we
////     want to minimize the number of headers included in .cu.cc files as NVCC
////     does not particularly like templates defined in ABSL.
////
extern template class Adaptor<BF16xBF16ToBF16>;
extern template class DeviceKernel<BF16xBF16ToBF16>;
//
////===----------------------------------------------------------------------===//
//// CUTLASS kernel arguments packing
////===----------------------------------------------------------------------===//
//
using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

template <typename Dim>
static Dim As(Dim3 dim3) {
  return Dim(dim3.x, dim3.y, dim3.z);
}

template <typename Dim>
static std::optional<Dim> As(std::optional<Dim3> dim3) {
  if (dim3.has_value()) return Dim(dim3->x, dim3->y, dim3->z);
  return std::nullopt;
}

template <typename Tag>
KernelArgsPacking ArgsPacking(int32_t m, int32_t n, int32_t k,
                              int32_t device_sms, Adaptor<Tag> adaptor) {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;

  // TODO(ezhulenev): CUTLASS kernel Params struct not necessarily trivially
  // destructible or even trivially copyable, we have to own the life time of an
  // object constructed in the storage. For now we ignore it, and it's textbook
  // definition of UB, but for CUTLASS kernels we use today it's perfectly safe.
  struct Params {
    std::byte storage[1024];
//#if defined(_MSC_VER)
//    alignas(64) std::byte storage[1024];
//#else
//    alignas(128) std::byte storage[1024];
//#endif
  };

  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

    Arguments arguments {
      const_cast<const void*>(mem_args->device_memory_ptr(0)),
      const_cast<const void*>(mem_args->device_memory_ptr(1)),
      const_cast<void*>(mem_args->device_memory_ptr(2)),
      m, n, k, 0, 0, 0, 1
    };

    Params params;
    adaptor.Initialize(&params, arguments, device_sms);

    return se::PackKernelArgs<Params>(
        args.number_of_shared_bytes(), params);
  };
}
//===----------------------------------------------------------------------===//

template <typename Tag>
static CustomKernel Load(std::string name,
                         int32_t m, int32_t n, int32_t k,
                         const se::DeviceDescription& device,
                         Adaptor<Tag> adaptor = {},
                         DeviceKernel<Tag> kernel = {}) {
  // Get the dispatch grid size and shared memory requirements.
  auto block_dim = As<se::BlockDim>(adaptor.BlockDim(m, n, k));
  auto thread_dim = As<se::ThreadDim>(adaptor.ThreadDim());
  // todo(esjoblom): ck uses static lds?
  auto shared_memory_bytes = 0; //adaptor.SharedMemoryBytes();

  auto packing = ArgsPacking<Tag>(m, n, k, device.core_count(), adaptor);

  se::MultiKernelLoaderSpec spec(/*arity=*/1, std::move(packing));
  spec.AddInProcessSymbol(kernel.symbol(), name);

  return CustomKernel(std::move(name), std::move(spec), block_dim, thread_dim,
                      shared_memory_bytes);
}

absl::StatusOr<std::vector<CustomKernel>> GetCkGemmKernels(
    std::string name, PrimitiveType dot_type, PrimitiveType lhs_type,
    PrimitiveType rhs_type, int32_t m, int32_t n, int32_t k,
    const se::DeviceDescription& device) {
  // Lookup table for supported kernels.
  // LHS_TYPE, RHS_TYPE, DOT_TYPE -> [kernel]
  absl::flat_hash_map<std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>,
                      std::vector<CustomKernel>>
      kernels = {
          {{BF16, BF16, BF16},
           {Load<BF16xBF16ToBF16>(name, m, n, k, device)}}};

  auto loaded_kernels = kernels.find({lhs_type, rhs_type, dot_type});
  if (loaded_kernels != kernels.end()) {
    return loaded_kernels->second;
  } else {
    std::string kernel_name = PrimitiveType_Name(lhs_type) + "x" +
                              PrimitiveType_Name(rhs_type) + "To" +
                              PrimitiveType_Name(dot_type);
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported CK gemm data type for kernel: ", kernel_name));
  }
}

}  // namespace xla::gpu::kernel::gemm_universal
