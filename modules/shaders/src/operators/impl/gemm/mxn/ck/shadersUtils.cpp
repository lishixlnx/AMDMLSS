/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"
#include "../../utils.hpp"

#include "gfx1201/fp32/shadersBin.hpp"
#include "gfx1201/fp32/DecisionTree.hpp"

namespace ck_gfx1201 = mlss::gemm::mxn::ck::fp32::gfx1201;

using mlss::gemm::utils::GenericGemmParams;
using mlss::gemm::utils::buildGemmParams;

namespace mlss::gemm::mxn::ck
{

    namespace
    {

        // Tile geometry comes from Gemm2d_FP32 (mirrors dxcp). We only
        // expose K-alignment / minimum K here because they are part of
        // adapter-side admission control rather than per-shader geometry.
        constexpr std::uint32_t kAlignK = 32u;
        constexpr std::uint32_t kMinK   = 64u;

        bool isSkipShape(const GenericGemmParams& params)
        {
            // BERT-like shapes that the decision tree mis-classifies as 'use this kernel'.
            // Mirrors dxcp::IsSkipCases.
            constexpr std::array<std::array<std::uint32_t, 3u>, 5u> kSkipShapes = {{
                {{384u, 1024u, 1024u}},
                {{384u, 4096u, 1024u}},
                {{384u, 1024u, 4096u}},
                {{384u,   64u,  384u}},
                {{384u,    2u, 1024u}},
            }};

            for (const auto& shape : kSkipShapes)
            {
                if ((params.m == shape[0u]) && (params.n == shape[1u]) && (params.k == shape[2u]))
                {
                    return true;
                }
            }
            return false;
        }

        bool isAdapterSupported(const GfxIpTriple& ip, const GenericGemmParams& params)
        {
            if (!isGfx12(ip))
            {
                return false;
            }

            // CK GEMM only supports NN.
            if (params.transA || params.transB)
            {
                return false;
            }

            // FP32 input + FP32 output + FP32 precision only.
            if (params.dataType != DataTypeFlags::FLOAT32)
            {
                return false;
            }
            if ((params.precision != PrecisionFlags::COUNT) &&
                (params.precision != PrecisionFlags::FLOAT32))
            {
                return false;
            }

            // Kernel layout requires aligned dimensions.
            if (((params.m % Gemm2d_FP32[0x01u]) != 0u) ||
                ((params.n % Gemm2d_FP32[0x02u]) != 0u) ||
                ((params.k % kAlignK)            != 0u))
            {
                return false;
            }

            // And minimum sizes.
            if ((params.m < Gemm2d_FP32[0x01u]) ||
                (params.n < Gemm2d_FP32[0x02u]) ||
                (params.k < kMinK))
            {
                return false;
            }

            // Alpha is fused at 1.0; non-default alpha falls back to MSFT.
            if (params.alpha != 1.0f)
            {
                return false;
            }

            // No bias, no batching, no activation in this binary.
            if (params.hasC || (params.batch > 1u))
            {
                return false;
            }
            if ((params.activation != ActivationFunctionFlags::COUNT) &&
                (params.activation != ActivationFunctionFlags::IDENTITY))
            {
                return false;
            }

            return true;
        }

        bool fullSupport(const GenericGemmParams& params)
        {
            const std::array<float, 3u> features = {
                static_cast<float>(params.m),
                static_cast<float>(params.n),
                static_cast<float>(params.k)
            };

            const auto predicted = static_cast<CKGemmShader>(
                predictHelper(ck_gfx1201::Navi48CKGemmFp32Tree, std::span{features}));

            if (predicted == CKGemmShader::ShaderSkipMetaCmd)
            {
                return false;
            }

            return !isSkipShape(params);
        }

        Binaries makeBinaries(const GenericGemmParams& params, const GfxIpTriple& gfxip)
        {
            const std::uint32_t blockCountX =
                integer_divide_ceil(params.m, Gemm2d_FP32[0x01u]) *
                integer_divide_ceil(params.n, Gemm2d_FP32[0x02u]);

            MLSSdim3 grid{blockCountX, 1u, 1u};
            MLSSdim3 blocks{Gemm2d_FP32[0x04u],
                            Gemm2d_FP32[0x05u],
                            Gemm2d_FP32[0x06u]};

            Binaries binaries;

            // 1) Relocatable variant: the original ELF blob from the archive.
            auto relocDescriptor = make_shader_descriptor(
                std::span<const std::uint8_t>(ck_gfx1201::gemm_add_fp32_add_dl_nn_gfx1201),
                "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto relocBlob = make_binary_blob(relocDescriptor);
            if (relocBlob)
            {
                *relocBlob = ck_gemm_mxn_ARGS_CONSTANTS;
                relocBlob->m_constants.assign(Gemm2d_FP32.begin(), Gemm2d_FP32.end());
                relocBlob->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*relocBlob));
            }

            // 2) Non-relocatable variant: link against the runtime target.
            //    The CK FP32 binary is compiled for gfx1201 (sole CK arch).
            constexpr GfxIpTriple kSourceArch{0x0Cu, 0x00u, 0x01u};
            auto nonRelocResult = getNonRelocatable(relocDescriptor.m_binary, kSourceArch, gfxip);
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
            nonRelocBlob = ck_gemm_mxn_ARGS_CONSTANTS;
            nonRelocBlob.m_constants.assign(Gemm2d_FP32.begin(), Gemm2d_FP32.end());
            nonRelocBlob.setGridBlocks(grid, blocks);
            binaries.addBlob(std::move(nonRelocBlob));

            return binaries;
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
        const bool isFullySupported = isSupported && fullSupport(params);

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

        return makeBinaries(params, ip);
    }

} // namespace mlss::gemm::mxn::ck
