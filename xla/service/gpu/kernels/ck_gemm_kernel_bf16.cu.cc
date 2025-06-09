//#include <cstring>
//#include <iostream>
//#include <ostream>
//#include <string>
//#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "xla/service/gpu/kernels/ck_gemm_adaptor.cu.h"

namespace xla::gpu::kernel::gemm_universal {

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

using ADataType = ck_tile::bf16_t;
using BDataType = ck_tile::bf16_t;
using AccDataType = float;
using CDataType = ck_tile::bf16_t;

constexpr bool kPadM = false;
constexpr bool kPadN = false;
constexpr bool kPadK = false;

constexpr int kBlockPerCu = 1;

// This part comes from the Codegen
constexpr ck_tile::index_t M_Tile = 128;
constexpr ck_tile::index_t N_Tile = 128;
constexpr ck_tile::index_t K_Tile = 32;

constexpr ck_tile::index_t M_Warp = 2;
constexpr ck_tile::index_t N_Warp = 2;
constexpr ck_tile::index_t K_Warp = 1;

constexpr ck_tile::index_t M_Warp_Tile = 32;
constexpr ck_tile::index_t N_Warp_Tile = 32;
constexpr ck_tile::index_t K_Warp_Tile = 16;

using CodegenGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                           ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                           ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

using CodegenGemmTraits =
    ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;
using CodegenPipelineProblem = ck_tile::
    GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;
using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;
using GemmEpilogue        = ck_tile::CShuffleEpilogue<
    ck_tile::CShuffleEpilogueProblem<AccDataType,
                                     CDataType,
                                     CLayout,
                                     CodegenPipelineProblem::kBlockSize,
                                     TilePartitioner::MPerBlock,
                                     TilePartitioner::NPerBlock,
                                     M_Warp,
                                     N_Warp,
                                     M_Warp_Tile,
                                     N_Warp_Tile,
                                     K_Warp_Tile,
                                     CodegenPipelineProblem::TransposeC>>;
using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;


XLA_GPU_DEFINE_CK_GEMM_TRAITS(BF16xBF16ToBF16, Kernel);

template class Adaptor<BF16xBF16ToBF16>;
template class DeviceKernel<BF16xBF16ToBF16>;

}  // namespace xla::gpu::kernel::gemm_universal
