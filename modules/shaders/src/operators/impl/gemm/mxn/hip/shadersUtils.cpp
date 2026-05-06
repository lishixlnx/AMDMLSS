/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "../../utils.hpp"
#include "../../../conv/1x1/hip/wmma/shadersConstants.hpp"

#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

namespace gfx1100        = mlss::gemm::mxn::hip::fp16::gfx1100;
namespace gfx1150        = mlss::gemm::mxn::hip::fp16::gfx1150;
namespace gfx1201        = mlss::gemm::mxn::hip::fp16::gfx1201;
namespace gfx1201_noact  = mlss::gemm::mxn::hip::fp16::gfx1201::noActivations;
namespace conv_trees     = mlss::conv::one_by_one::hip::wmma::fp16;

using mlss::gemm::utils::GenericGemmParams;
using mlss::gemm::utils::buildGemmParams;
using mlss::gemm::utils::chooseTuning;

namespace mlss::gemm::mxn::hip
{

    namespace
    {

        constexpr std::uint32_t kMinDim         = 16u;
        constexpr std::uint32_t kEvenMask       = 1u;
        constexpr std::uint32_t kBlockSizeMain  = 256u;
        constexpr std::uint32_t kGfx12MaxBlocks = 0xFFFFu;

        const std::array<std::uint32_t, 7u>& getShaderConstants(HipGemmShader shader)
        {
            switch (shader)
            {
                case HipGemmShader::Shader16x16x16WMMA_NN:    return Gemm2d_16x16x16NN;
                case HipGemmShader::Shader16x128x16WMMA_NN:   return Gemm2d_16x128x16NN;
                case HipGemmShader::Shader16x256x16WMMA_NN:   return Gemm2d_16x256x16NN;
                case HipGemmShader::Shader64x128x32WMMA_NN:   return Gemm2d_64x128x32NN;
                case HipGemmShader::Shader128x128x16WMMA_NN:  return Gemm2d_128x128x16NN;
                case HipGemmShader::Shader128x64x32WMMA_NN:   return Gemm2d_128x64x32NN;
                case HipGemmShader::Shader256x16x16WMMA_NN:   return Gemm2d_256x16x16NN;
                case HipGemmShader::Shader64x128x32WMMA_NT:   return Gemm2d_64x128x32NT;
                default:                                      return Gemm2d_16x16x16NN;
            }
        }

        bool isAdapterSupported(const GfxIpTriple& ip, const GenericGemmParams& params)
        {
            if (!isGfx110x(ip) && !isGfx115x(ip) && !isGfx120x(ip))
            {
                return false;
            }

            // Only fp16 input with fp16 output is supported (matches dxcp).
            if (params.dataType != DataTypeFlags::FLOAT16)
            {
                return false;
            }

            if ((params.precision != PrecisionFlags::COUNT) &&
                (params.precision == PrecisionFlags::FLOAT32))
            {
                return false;
            }

            // Tensor A column-major (transA) is not supported.
            if (params.transA)
            {
                return false;
            }

            // Minimum dimensions and N/K must be even (kernel layout requirement).
            if ((params.m < kMinDim) || (params.n < kMinDim) || (params.k < kMinDim))
            {
                return false;
            }
            if (((params.n & kEvenMask) != 0u) || ((params.k & kEvenMask) != 0u))
            {
                return false;
            }

            // The transposed-B path only has the 64x128x32 NT shader.
            if (params.transB && ((params.m < 64u) || (params.n < 128u) || (params.k < 32u)))
            {
                return false;
            }

            // We support every element-wise activation (the "MAX" family is excluded).
            if ((params.activation != ActivationFunctionFlags::COUNT) &&
                ((params.activation == ActivationFunctionFlags::HARDMAX)     ||
                 (params.activation == ActivationFunctionFlags::LOG_SOFTMAX) ||
                 (params.activation == ActivationFunctionFlags::SOFTMAX)))
            {
                return false;
            }

            return true;
        }

        // Decision-tree based shader selection. The Gemm trees are shared with
        // the conv 1x1 backend so we reuse them directly to avoid duplication.
        HipGemmShader chooseNNShader(const GfxIpTriple& gfxip, const GenericGemmParams& params)
        {
            const std::array<float, 3u> features = {
                static_cast<float>(params.m),
                static_cast<float>(params.n),
                static_cast<float>(params.k)
            };

            auto shader = HipGemmShader::Shader16x16x16WMMA_NN;

            if (gfxip == IP_GFX1100)
            {
                shader = static_cast<HipGemmShader>(
                    predictHelper(conv_trees::Navi31HipGemmFp16Tree, std::span{features}));
            }
            else if (gfxip == IP_GFX1101)
            {
                shader = static_cast<HipGemmShader>(
                    predictHelper(conv_trees::Navi32HipGemmFp16Tree, std::span{features}));
            }
            else if (gfxip == IP_GFX1102 || gfxip == IP_GFX1103)
            {
                shader = static_cast<HipGemmShader>(
                    predictHelper(conv_trees::Navi33HipGemmFp16Tree, std::span{features}));
            }
            else if (isGfx115x(gfxip))
            {
                shader = static_cast<HipGemmShader>(
                    predictHelper(conv_trees::StrixHipGemmFp16Tree, std::span{features}));
            }
            else if (gfxip == IP_GFX1200 || gfxip == IP_GFX1201)
            {
                constexpr std::array<std::int32_t, 7u> Navi48Labels = {
                    static_cast<std::int32_t>(HipGemmShader::Shader16x16x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader16x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader16x256x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader64x128x32WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader128x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader128x64x32WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader256x16x16WMMA_NN)
                };
                shader = static_cast<HipGemmShader>(
                    predictHelper(conv_trees::Navi48HipGemmFp16Tree, std::span{features},
                                  std::span{Navi48Labels}));
            }
            else if (gfxip == IP_GFX1210 || gfxip == IP_GFX1211)
            {
                constexpr std::array<std::int32_t, 7u> Navi44Labels = {
                    static_cast<std::int32_t>(HipGemmShader::Shader16x16x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader16x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader16x256x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader64x128x32WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader128x128x16WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader128x64x32WMMA_NN),
                    static_cast<std::int32_t>(HipGemmShader::Shader256x16x16WMMA_NN)
                };
                shader = static_cast<HipGemmShader>(
                    predictHelper(conv_trees::Navi44HipGemmFp16Tree, std::span{features},
                                  std::span{Navi44Labels}));
            }

            // Fall back to the 16x16 catch-all if the predicted tile does not
            // fit the problem size. Mirrors dxcp::ChooseNNShader.
            const auto& constants = getShaderConstants(shader);
            if ((params.m < constants[0x01u]) ||
                (params.n < constants[0x02u]) ||
                (params.k < constants[0x03u]))
            {
                shader = HipGemmShader::Shader16x16x16WMMA_NN;
            }

            return shader;
        }

        // Map runtime target IP to the GFX target the archived ELF was
        // compiled for. Mirrors conv1x1/hip/wmma::sourceArchForTarget.
        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x00u) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x05u) return {0x0Bu, 0x05u, 0x00u};
            if (gfxip.major == 0x0Cu)                         return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        template <std::size_t N>
        Binaries makeKernelBinaries(const std::array<std::uint8_t, N>& data,
                                    const std::array<std::uint32_t, 7u>& constants,
                                    const GenericGemmParams& params,
                                    const GfxIpTriple& gfxip)
        {
            std::uint32_t blockCountX = integer_divide_ceil(params.m, constants[0x01u]);
            std::uint32_t blockCountY = integer_divide_ceil(params.n, constants[0x02u]);

            if (isGfx12(gfxip))
            {
                blockCountY = std::min(blockCountY, kGfx12MaxBlocks);
            }

            MLSSdim3 grid{blockCountX, blockCountY, params.batch};
            MLSSdim3 blocks{constants[0x04u] * constants[0x05u] * constants[0x06u],
                            1u, 1u};

            Binaries binaries;

            // 1) Relocatable variant: the original ELF blob from the archive.
            auto relocDescriptor = make_shader_descriptor(
                std::span<const std::uint8_t>(data), "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto relocBlob = make_binary_blob(relocDescriptor);
            if (relocBlob)
            {
                *relocBlob = hip_gemm_mxn_ARGS_CONSTANTS;
                relocBlob->m_constants.assign(constants.begin(), constants.end());
                relocBlob->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*relocBlob));
            }

            // 2) Non-relocatable variant: link the relocatable ELF against
            //    the runtime target.
            const auto sourceArch = sourceArchForTarget(gfxip);
            auto nonRelocResult = getNonRelocatable(relocDescriptor.m_binary, sourceArch, gfxip);
            if (!nonRelocResult.has_value())
            {
                return binaries;
            }

            // Take ownership of the linked bytes inside the Blob so the
            // pointer remains valid past this function (default
            // make_binary_blob would store a raw pointer into the source
            // span, which would dangle once the local goes out of scope).
            auto nonRelocBytes = std::move(nonRelocResult).value();
            const std::span<const std::uint8_t> nonRelocSpan(nonRelocBytes.data(), nonRelocBytes.size());

            const std::string nonRelocName =
                getKernelName(nonRelocSpan).value_or(std::string{});

            Binaries::Blob nonRelocBlob{
                nonRelocBytes.data(),
                nonRelocBytes.size(),
                static_cast<std::uint32_t>(BinaryTypeFlags::ELF),
                0u,
                nonRelocName};
            nonRelocBlob.setOwnedBinary(std::move(nonRelocBytes));
            nonRelocBlob = hip_gemm_mxn_ARGS_CONSTANTS;
            nonRelocBlob.m_constants.assign(constants.begin(), constants.end());
            nonRelocBlob.setGridBlocks(grid, blocks);
            binaries.addBlob(std::move(nonRelocBlob));

            return binaries;
        }

        std::expected<Binaries, std::error_code> buildBinariesGfx1100(
            const GenericGemmParams& params,
            const GfxIpTriple& gfxip,
            HipGemmShader shader)
        {
            const auto& constants = getShaderConstants(shader);
            switch (shader)
            {
                case HipGemmShader::Shader64x128x32WMMA_NT:
                    return makeKernelBinaries(gfx1100::gemm_add_fp16_64x128x32_add_nt_gfx1100, constants, params, gfxip);
                case HipGemmShader::Shader64x128x32WMMA_NN:
                    return makeKernelBinaries(gfx1100::gemm_add_fp16_64x128x32_add_nn_gfx1100, constants, params, gfxip);
                case HipGemmShader::Shader16x256x16WMMA_NN:
                    return makeKernelBinaries(gfx1100::gemm_add_fp16_16x256x16_add_1_2_nn_gfx1100, constants, params, gfxip);
                case HipGemmShader::Shader16x128x16WMMA_NN:
                    return makeKernelBinaries(gfx1100::gemm_add_fp16_16x128x16_add_nn_gfx1100, constants, params, gfxip);
                case HipGemmShader::Shader16x16x16WMMA_NN:
                    return makeKernelBinaries(gfx1100::gemm_add_fp16_16x16x16_add_nn_gfx1100, constants, params, gfxip);
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
            }
        }

        std::expected<Binaries, std::error_code> buildBinariesGfx1150(
            const GenericGemmParams& params,
            const GfxIpTriple& gfxip,
            HipGemmShader shader)
        {
            const auto& constants = getShaderConstants(shader);
            switch (shader)
            {
                case HipGemmShader::Shader64x128x32WMMA_NT:
                    return makeKernelBinaries(gfx1150::gemm_add_fp16_64x128x32_add_nt_gfx1150, constants, params, gfxip);
                case HipGemmShader::Shader64x128x32WMMA_NN:
                    return makeKernelBinaries(gfx1150::gemm_add_fp16_64x128x32_add_nn_gfx1150, constants, params, gfxip);
                case HipGemmShader::Shader16x256x16WMMA_NN:
                    return makeKernelBinaries(gfx1150::gemm_add_fp16_16x256x16_add_1_2_nn_gfx1150, constants, params, gfxip);
                case HipGemmShader::Shader16x128x16WMMA_NN:
                    return makeKernelBinaries(gfx1150::gemm_add_fp16_16x128x16_add_nn_gfx1150, constants, params, gfxip);
                case HipGemmShader::Shader16x16x16WMMA_NN:
                    return makeKernelBinaries(gfx1150::gemm_add_fp16_16x16x16_add_nn_gfx1150, constants, params, gfxip);
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
            }
        }

        std::expected<Binaries, std::error_code> buildBinariesGfx1201(
            const GenericGemmParams& params,
            const GfxIpTriple& gfxip,
            HipGemmShader shader)
        {
            const bool useNoActivation = (params.activation == ActivationFunctionFlags::IDENTITY);
            const auto& constants = getShaderConstants(shader);

            if (shader == HipGemmShader::Shader64x128x32WMMA_NT)
            {
                return useNoActivation
                    ? makeKernelBinaries(gfx1201_noact::gemm_add_fp16_64x128x32_add_2_4_nt_gfx1201, constants, params, gfxip)
                    : makeKernelBinaries(gfx1201::gemm_add_fp16_64x128x32_add_nt_gfx1201, constants, params, gfxip);
            }

            if (useNoActivation)
            {
                switch (shader)
                {
                    case HipGemmShader::Shader128x128x16WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_128x128x16_add_nn_gfx1201, constants, params, gfxip);
                    case HipGemmShader::Shader128x64x32WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_128x64x32_add_tr_nn_gfx1201, constants, params, gfxip);
                    case HipGemmShader::Shader64x128x32WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_64x128x32_add_tr_nn_gfx1201, constants, params, gfxip);
                    case HipGemmShader::Shader256x16x16WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_256x16x16_add_2_1_nn_gfx1201, constants, params, gfxip);
                    case HipGemmShader::Shader16x256x16WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_16x256x16_add_1_2_nn_gfx1201, constants, params, gfxip);
                    case HipGemmShader::Shader16x128x16WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_16x128x16_add_nn_gfx1201, constants, params, gfxip);
                    case HipGemmShader::Shader16x16x16WMMA_NN:
                        return makeKernelBinaries(gfx1201_noact::gemm_add_fp16_16x16x16_add_native_nn_gfx1201, constants, params, gfxip);
                    default:
                        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
                }
            }

            switch (shader)
            {
                case HipGemmShader::Shader128x128x16WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_128x128x16_add_nn_gfx1201, constants, params, gfxip);
                case HipGemmShader::Shader128x64x32WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_128x64x32_add_nn_gfx1201, constants, params, gfxip);
                case HipGemmShader::Shader64x128x32WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_64x128x32_add_nn_gfx1201, constants, params, gfxip);
                case HipGemmShader::Shader256x16x16WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_256x16x16_add_nn_gfx1201, constants, params, gfxip);
                case HipGemmShader::Shader16x256x16WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_16x256x16_add_nn_gfx1201, constants, params, gfxip);
                case HipGemmShader::Shader16x128x16WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_16x128x16_add_nn_gfx1201, constants, params, gfxip);
                case HipGemmShader::Shader16x16x16WMMA_NN:
                    return makeKernelBinaries(gfx1201::gemm_add_fp16_16x16x16_add_nn_gfx1201, constants, params, gfxip);
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
            }
        }

    } // anonymous namespace

    mlss::op::utils::MetaCmdCaps isShadersAvailable(const GfxIpTriple& ip,
                                                    const std::vector<Attribute>& attr,
                                                    const void* cstmStruct)
    {
        using mlss::op::utils::MetaCmdCaps;

        const auto params = cstmStruct
            ? *static_cast<const GenericGemmParams*>(cstmStruct)
            : buildGemmParams(attr);

        const bool isSupported     = isAdapterSupported(ip, params);
        const bool isFullySupported = isSupported && chooseTuning(ip, params);

        MetaCmdCaps caps{.values = 0x00000000u};
        caps.support     = isSupported       ? 0x00000001u : 0x00000000u;
        caps.fullSupport = isFullySupported  ? 0x00000001u : 0x00000000u;
        return caps;
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip,
                                                            const std::vector<Attribute>& attr,
                                                            const void* cstmStruct)
    {
        const auto params = cstmStruct
            ? *static_cast<const GenericGemmParams*>(cstmStruct)
            : buildGemmParams(attr);

        const auto caps = isShadersAvailable(ip, attr, &params);
        if (caps.support == 0u)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
        }

        const HipGemmShader shader = params.transB
            ? HipGemmShader::Shader64x128x32WMMA_NT
            : chooseNNShader(ip, params);

        if (isGfx110x(ip))
        {
            return buildBinariesGfx1100(params, ip, shader);
        }
        if (isGfx115x(ip))
        {
            return buildBinariesGfx1150(params, ip, shader);
        }
        if (isGfx120x(ip))
        {
            return buildBinariesGfx1201(params, ip, shader);
        }

        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
    }

} // namespace mlss::gemm::mxn::hip
