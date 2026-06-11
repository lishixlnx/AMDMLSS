/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
// Verify that the converted archive headers are stdlib-only and compile in
// isolation. Build with:
//   clang++ -std=c++23 -I .. -c verify_compile.cpp
// (run from tools/archive_normalize). No project headers are included.

#include "../../archive/conv/1x1/gfx1100/fp16/shadersBinReloc.hpp"
#include "../../archive/conv/1x1/wmma/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/conv/1x1/wmma/shadersConstants.hpp"
#include "../../archive/conv/dilated/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/conv/dw/hlsl/fp16/shadersIL.hpp"
#include "../../archive/conv/mxn/Misa/gfx1100/fp16/shadersBinReloc.hpp"
#include "../../archive/conv/mxn/Winograd/Base/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/conv/mxn/Winograd/Fury/gfx1100/fp16/shadersBinReloc.hpp"
#include "../../archive/conv/mxn/Winograd/Rage/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/gemm/1xn/hlsl/fp16/shadersIL.hpp"
#include "../../archive/gemm/mxn/ck/gfx1201/fp32/shadersBinReloc.hpp"
#include "../../archive/gemm/mxn/hip/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/gqa/wmma/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/mha/wmma/gfx1201/fp16/shadersBinReloc.hpp"
#include "../../archive/qgemm/mxn/hlsl/wmma/WZQ/128/64x64/gfx1201/fp16_u4/shadersIL.hpp"

#include <cstddef>

namespace
{

// One sentinel from each header to prove the symbols resolve and the data
// is constexpr-accessible.
constexpr std::size_t kProbe =
    archive::conv::one_by_one::gfx1100::fp16::gemm_add_fp16_misa_1x1_add_gfx1100.size()
  + archive::conv::dw::hlsl::fp16::conv_relu_add_fp16_dw_relu_add_general_hlsl.size()
  + archive::conv::one_by_one::wmma::Navi31HipGemmFp16Tree.m_leftChildren.size()
  + archive::conv::one_by_one::wmma::hip_conv_1x1_ARGS.size()
  + archive::qgemm::mxn::hlsl::wmma::wzq::q128::tile_64x64::gfx1201::fp16_u4::
        qgemm_fp16_u4_hlsl_wmma_wzq_q128_64x64_gfx1201.size();

static_assert(kProbe > 0u);

} // namespace

int main()
{
    return static_cast<int>(kProbe & 0x7Fu);
}
