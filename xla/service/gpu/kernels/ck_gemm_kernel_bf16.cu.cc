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

//struct GemmConfig {
//    // Compute friendly for Intrawave scheduler
//    static constexpr ck_tile::index_t M_Tile = 128;
//    static constexpr ck_tile::index_t N_Tile = 128;
//    static constexpr ck_tile::index_t K_Tile = 128;
//
//    static constexpr ck_tile::index_t M_Warp = 2;
//    static constexpr ck_tile::index_t N_Warp = 2;
//    static constexpr ck_tile::index_t K_Warp = 1;
//
//    static constexpr ck_tile::index_t M_Warp_Tile = 16;
//    static constexpr ck_tile::index_t N_Warp_Tile = 16;
//    static constexpr ck_tile::index_t K_Warp_Tile = 32;
//
//    static constexpr bool DoubleSmemBuffer = false;
//
//    static constexpr bool kPadM = false;
//    static constexpr bool kPadN = false;
//    static constexpr bool kPadK = false;
//
//    static constexpr bool PermuteA = false;
//    static constexpr bool PermuteB = false;
//
//    static constexpr bool TransposeC            = false;
//    static constexpr bool UseStructuredSparsity = false;
//
//    static constexpr int kBlockPerCu                         = 1;
//    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
//    static constexpr ck_tile::index_t TileParitionerM01      = 4;
//};
//
//using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
//using BLayout = ck_tile::tensor_layout::gemm::RowMajor;
//using CLayout = ck_tile::tensor_layout::gemm::RowMajor;
//
//using ADataType = ck_tile::bf16_t;
//using BDataType = ck_tile::bf16_t;
//using AccDataType = float;
//using CDataType = ck_tile::bf16_t;
//
//using GemmShape = ck_tile::TileGemmShape<
//    ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
//    ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
//    ck_tile::
//        sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
//
//using TilePartitioner =
//    ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
//                                               GemmConfig::TileParitionerGroupNum,
//                                               GemmConfig::TileParitionerM01>;
//
//using TGemmTraits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
//                                            GemmConfig::kPadN,
//                                            GemmConfig::kPadK,
//                                            ALayout,
//                                            BLayout,
//                                            CLayout>;
//using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
//                                                             GemmConfig::kPadN,
//                                                             GemmConfig::kPadK,
//                                                             ALayout,
//                                                             BLayout,
//                                                             CLayout,
//                                                             GemmConfig::TransposeC>;
//using GemmPipelineProblem =
//    ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, TGemmTraits>;
//
//using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>;
//
//constexpr ck_tile::index_t k_grain     = 1 * GemmConfig::K_Tile;
//constexpr ck_tile::index_t K_split     = 2048;
//constexpr ck_tile::index_t num_loop    = 64; //TilePartitioner::GetLoopNum(K_split);
////const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
//constexpr ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
//
////const auto Run =
////    [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
////constexpr bool has_hot_loop_v   = has_hot_loop.value;
////constexpr auto tail_number_v    = tail_number_.value;
//constexpr auto scheduler        = ck_tile::GemmPipelineScheduler::Intrawave;
//
//using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
//                                                                   BDataType,
//                                                                   AccDataType,
//                                                                   GemmShape,
//                                                                   GemmUniversalTraits,
//                                                                   scheduler,
//                                                                   false,
//                                                                   tail_num>;
//
//using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;
////using GemmEpilogue = ck_tile::CShuffleEpilogue<
////    ck_tile::CShuffleEpilogueProblem<ADataType,
////                                     BDataType,
////                                     AccDataType,
////                                     CDataType,
////                                     CLayout,
////                                     GemmPipelineProblem::kBlockSize,
////                                     TilePartitioner::MPerBlock,
////                                     TilePartitioner::NPerBlock,
////                                     GemmConfig::M_Warp,
////                                     GemmConfig::N_Warp,
////                                     GemmConfig::M_Warp_Tile,
////                                     GemmConfig::N_Warp_Tile,
////                                     GemmConfig::K_Warp_Tile,
////                                     UniversalGemmProblem::TransposeC>>;
//using GemmEpilogue = ck_tile::CShuffleEpilogue<
//    ck_tile::CShuffleEpilogueProblem<AccDataType,
//                                     CDataType,
//                                     CLayout,
//                                     GemmPipelineProblem::kBlockSize,
//                                     TilePartitioner::MPerBlock,
//                                     TilePartitioner::NPerBlock,
//                                     GemmConfig::M_Warp,
//                                     GemmConfig::N_Warp,
//                                     GemmConfig::M_Warp_Tile,
//                                     GemmConfig::N_Warp_Tile,
//                                     GemmConfig::K_Warp_Tile,
//                                     UniversalGemmProblem::TransposeC>>;
//using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

using ADataType = ck_tile::bf16_t;
using BDataType = ck_tile::bf16_t;
using AccDataType = float;
using CDataType = ck_tile::bf16_t;

constexpr bool kPadM = true;
constexpr bool kPadN = true;
constexpr bool kPadK = true;

constexpr int kBlockPerCu = 1;

// This part comes from the Codegen
constexpr ck_tile::index_t M_Tile = 128;
constexpr ck_tile::index_t N_Tile = 128;
constexpr ck_tile::index_t K_Tile = 64;

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
