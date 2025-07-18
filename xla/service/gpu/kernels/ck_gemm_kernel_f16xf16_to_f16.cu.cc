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

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "xla/service/gpu/kernels/ck_gemm_adaptor.cu.h"

namespace xla::gpu::kernel::gemm_universal {

//===----------------------------------------------------------------------===//
// FP16 Universal GEMM Kernel Configuration  
//===----------------------------------------------------------------------===//

constexpr ck_tile::index_t M_Tile = 128;
constexpr ck_tile::index_t N_Tile = 128;
constexpr ck_tile::index_t K_Tile = 32;

constexpr ck_tile::index_t M_Warp = 2;
constexpr ck_tile::index_t N_Warp = 2;
constexpr ck_tile::index_t K_Warp = 1;

constexpr ck_tile::index_t M_Warp_Tile = 32;
constexpr ck_tile::index_t N_Warp_Tile = 32;
constexpr ck_tile::index_t K_Warp_Tile = 8;

constexpr bool kPadM = true;
constexpr bool kPadN = true;
constexpr bool kPadK = true;
constexpr bool TransposeC = false;

constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
constexpr ck_tile::index_t TilePartitionerM01 = 4;

constexpr bool has_hot_loop = false;
constexpr ck_tile::TailNumber tail_number = ck_tile::TailNumber::One;
constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

using ADataType = ck_tile::half_t;
using BDataType = ck_tile::half_t;
using AccDataType = float;
using CDataType = ck_tile::half_t;

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::RowMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

//===----------------------------------------------------------------------===//
// Universal GEMM Kernel Assembly
//===----------------------------------------------------------------------===//

using GemmShape = ck_tile::TileGemmShape<
    ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
    ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
    ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
    GemmShape, TilePartitionerGroupNum, TilePartitionerM01>;

using TileGemmTraits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
    kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, TransposeC>;

using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
    ADataType, BDataType, AccDataType, GemmShape, TileGemmTraits>;

using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
    ADataType, BDataType, AccDataType, GemmShape, GemmUniversalTraits, 
    scheduler, has_hot_loop, tail_number>;

using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<
    UniversalGemmProblem, ck_tile::UniversalGemmPipelineAgBgCrPolicy>;

using GemmEpilogue = ck_tile::CShuffleEpilogue<
    ck_tile::CShuffleEpilogueProblem<
        AccDataType, CDataType, CLayout,
        GemmPipelineProblem::kBlockSize,
        TilePartitioner::MPerBlock,
        TilePartitioner::NPerBlock,
        M_Warp, N_Warp,
        M_Warp_Tile, N_Warp_Tile, K_Warp_Tile,
        UniversalGemmProblem::TransposeC>>;

using CkGemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

XLA_GPU_DEFINE_CK_GEMM_TRAITS(F16xF16ToF16, CkGemmKernel);

template class Adaptor<F16xF16ToF16>;
template class DeviceKernel<F16xF16ToF16>;

}  // namespace xla::gpu::kernel::gemm_universal
