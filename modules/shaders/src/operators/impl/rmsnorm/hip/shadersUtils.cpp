/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

#include "fp16/gfx1200/shadersBin.hpp"
#include "fp16/gfx1201/shadersBin.hpp"

namespace gfx1200 = mlss::rmsnorm::hip::fp16::gfx1200;
namespace gfx1201 = mlss::rmsnorm::hip::fp16::gfx1201;

namespace mlss::rmsnorm::hip
{

    using mlss::isGfx120x;

    namespace
    {

        struct RmsNormParams
        {
            std::uint32_t m{0};
            std::uint32_t n{0};
            std::uint32_t dataType{0};
        };

        RmsNormParams extractParams(const std::vector<Attribute>& attr)
        {
            RmsNormParams params;
            for (const auto& attribute : attr)
            {
                if (attribute.is(MLSS_ATTR_RMSNORM_M))
                    params.m = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_RMSNORM_N))
                    params.n = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_RMSNORM_DATATYPE))
                    params.dataType = attribute.value<std::uint32_t>();
            }
            return params;
        }

        bool isRmsNormSupported(const RmsNormParams& params, const GfxIpTriple& ip)
        {
            if (!isGfx120x(ip))
                return false;

            if (params.m == 0u || params.n == 0u)
                return false;

            if (params.dataType != MLSS_FLOAT16)
                return false;

            return true;
        }

        // Mirrors dxcp dispatch logic from DdiMetaCmdRmsNorm::Execute.
        HipRmsNormShader chooseShader(const RmsNormParams& params)
        {
            const std::uint32_t m = params.m;
            if (m == 1u)
                return HipRmsNormShader::ShaderBm1Bn128;
            if (m < 16u)
                return ((m & 1u) == 0u) ? HipRmsNormShader::ShaderBm2Bn128 : HipRmsNormShader::ShaderBm2Bn128Pad;
            return ((m % 8u) == 0u) ? HipRmsNormShader::ShaderBm8Bn128 : HipRmsNormShader::ShaderBm8Bn128Pad;
        }

        const std::array<std::uint32_t, 5u>& getShaderConstants(HipRmsNormShader shader)
        {
            switch (shader)
            {
                case HipRmsNormShader::ShaderBm1Bn128:    return RmsNorm_Bm1_Bn128_CONSTANTS;
                case HipRmsNormShader::ShaderBm2Bn128:    return RmsNorm_Bm2_Bn128_CONSTANTS;
                case HipRmsNormShader::ShaderBm2Bn128Pad: return RmsNorm_Bm2_Bn128Pad_CONSTANTS;
                case HipRmsNormShader::ShaderBm8Bn128:    return RmsNorm_Bm8_Bn128_CONSTANTS;
                case HipRmsNormShader::ShaderBm8Bn128Pad: return RmsNorm_Bm8_Bn128Pad_CONSTANTS;
                default:                                  return RmsNorm_Bm1_Bn128_CONSTANTS;
            }
        }

        // Map runtime target IP to the GFX target the archived ELF was compiled for.
        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Cu && gfxip.minor == 0x00u) return {0x0Cu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Cu)                         return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        template <std::size_t N>
        Binaries makeKernelBinaries(const std::array<std::uint8_t, N>& data,
                                    const std::array<std::uint32_t, 5u>& constants,
                                    const RmsNormParams& params,
                                    const GfxIpTriple& gfxip)
        {
            // Dispatch: one block per row (m blocks), blockM rows per block.
            // Grid = (ceil(m / blockM), 1, 1), blocks = (threadX, 1, 1).
            const std::uint32_t blockM = constants[3u];
            const std::uint32_t threadX = constants[0u];

            const std::uint32_t blockCountX = integer_divide_ceil(params.m, blockM);
            MLSSdim3 grid{blockCountX, 1u, 1u};
            MLSSdim3 blocks{threadX, 1u, 1u};

            Binaries binaries;

            // 1) Relocatable variant.
            auto relocDescriptor = make_shader_descriptor(
                std::span<const std::uint8_t>(data), "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto relocBlob = make_binary_blob(relocDescriptor);
            if (relocBlob)
            {
                *relocBlob = hip_rmsnorm_ARGS_CONSTANTS;
                relocBlob->m_constants.assign(constants.begin(), constants.end());
                relocBlob->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*relocBlob));
            }

            // 2) Non-relocatable variant.
            const auto sourceArch = sourceArchForTarget(gfxip);
            auto nonRelocResult = getNonRelocatable(relocDescriptor.m_binary, sourceArch, gfxip);
            if (!nonRelocResult.has_value())
                return binaries;

            auto nonRelocBytes = std::move(nonRelocResult).value();
            const std::span<const std::uint8_t> nonRelocSpan(nonRelocBytes.data(), nonRelocBytes.size());

            const std::string nonRelocName = getKernelName(nonRelocSpan).value_or(std::string{});

            Binaries::Blob nonRelocBlob{
                nonRelocBytes.data(),
                nonRelocBytes.size(),
                static_cast<std::uint32_t>(BinaryTypeFlags::ELF),
                0u,
                nonRelocName};
            nonRelocBlob.setOwnedBinary(std::move(nonRelocBytes));
            nonRelocBlob = hip_rmsnorm_ARGS_CONSTANTS;
            nonRelocBlob.m_constants.assign(constants.begin(), constants.end());
            nonRelocBlob.setGridBlocks(grid, blocks);
            binaries.addBlob(std::move(nonRelocBlob));

            return binaries;
        }

    } // anonymous namespace

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;
        const auto params = extractParams(attr);
        return isRmsNormSupported(params, ip);
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;

        if (!isShadersAvailable(ip, attr, nullptr))
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));

        const auto params = extractParams(attr);
        const auto shader = chooseShader(params);

        // gfx1200 = Navi44 (minor==0x00), gfx1201 = Navi48 (minor==0x01).
        const auto& constants = getShaderConstants(shader);
        if (ip.minor == 0x00u)
        {
            switch (shader)
            {
                case HipRmsNormShader::ShaderBm1Bn128:
                    return makeKernelBinaries(gfx1200::RmsNorm_Fp16_Bm1_Bn128_Gfx1200, constants, params, ip);
                case HipRmsNormShader::ShaderBm2Bn128:
                    return makeKernelBinaries(gfx1200::RmsNorm_Fp16_Bm2_Bn128_Gfx1200, constants, params, ip);
                case HipRmsNormShader::ShaderBm2Bn128Pad:
                    return makeKernelBinaries(gfx1200::RmsNorm_Fp16_Bm2_Bn128_Pad_Gfx1200, constants, params, ip);
                case HipRmsNormShader::ShaderBm8Bn128:
                    return makeKernelBinaries(gfx1200::RmsNorm_Fp16_Bm8_Bn128_Gfx1200, constants, params, ip);
                case HipRmsNormShader::ShaderBm8Bn128Pad:
                    return makeKernelBinaries(gfx1200::RmsNorm_Fp16_Bm8_Bn128_Pad_Gfx1200, constants, params, ip);
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
            }
        }

        // gfx1201 (Navi48) and future minor variants
        switch (shader)
        {
            case HipRmsNormShader::ShaderBm1Bn128:
                return makeKernelBinaries(gfx1201::RmsNorm_Fp16_Bm1_Bn128_Gfx1201, constants, params, ip);
            case HipRmsNormShader::ShaderBm2Bn128:
                return makeKernelBinaries(gfx1201::RmsNorm_Fp16_Bm2_Bn128_Gfx1201, constants, params, ip);
            case HipRmsNormShader::ShaderBm2Bn128Pad:
                return makeKernelBinaries(gfx1201::RmsNorm_Fp16_Bm2_Bn128_Pad_Gfx1201, constants, params, ip);
            case HipRmsNormShader::ShaderBm8Bn128:
                return makeKernelBinaries(gfx1201::RmsNorm_Fp16_Bm8_Bn128_Gfx1201, constants, params, ip);
            case HipRmsNormShader::ShaderBm8Bn128Pad:
                return makeKernelBinaries(gfx1201::RmsNorm_Fp16_Bm8_Bn128_Pad_Gfx1201, constants, params, ip);
            default:
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
        }
    }

} // namespace mlss::rmsnorm::hip
