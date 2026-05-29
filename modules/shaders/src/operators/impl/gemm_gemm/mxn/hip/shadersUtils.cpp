/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

#include "fp16/gfx1200/shadersBin.hpp"
#include "fp16/gfx1201/shadersBin.hpp"
#include "w8a16/gfx1200/shadersBin.hpp"
#include "w8a16/gfx1201/shadersBin.hpp"

namespace fp16_1200 = mlss::gemm_gemm::mxn::hip::fp16::gfx1200;
namespace fp16_1201 = mlss::gemm_gemm::mxn::hip::fp16::gfx1201;
namespace w8a16_1200 = mlss::gemm_gemm::mxn::hip::w8a16::gfx1200;
namespace w8a16_1201 = mlss::gemm_gemm::mxn::hip::w8a16::gfx1201;

namespace mlss::gemm_gemm::mxn::hip
{

    using mlss::isGfx120x;

    namespace
    {

        enum class GemmGemmDtype { Fp16, W8A16PerTensor };

        struct GemmGemmParams
        {
            std::uint32_t m{0};
            std::uint32_t n{0};
            std::uint32_t k{0};
            std::uint32_t l{0};   // intermediate dimension (A×B0 → M×L, A×B1 → M×N)
            std::uint32_t batch{1};
            std::uint32_t dataType{0};
            std::uint32_t quantDataType{0};
            bool transb0{false};
            bool transb1{false};
        };

        GemmGemmParams extractParams(const std::vector<Attribute>& attr)
        {
            GemmGemmParams params;
            for (const auto& attribute : attr)
            {
                if (attribute.is(MLSS_ATTR_GEMMGEMM_M))
                    params.m = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_N))
                    params.n = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_K))
                    params.k = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_L))
                    params.l = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_BATCH))
                    params.batch = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_DATATYPE))
                    params.dataType = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_QUANTDATATYPE))
                    params.quantDataType = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_TRANSB0))
                    params.transb0 = attribute.value<bool>();
                else if (attribute.is(MLSS_ATTR_GEMMGEMM_TRANSB1))
                    params.transb1 = attribute.value<bool>();
            }
            return params;
        }

        GemmGemmDtype chooseDtype(const GemmGemmParams& params)
        {
            if (params.quantDataType == MLSS_INT8)
                return GemmGemmDtype::W8A16PerTensor;
            return GemmGemmDtype::Fp16;
        }

        bool isGemmGemmSupported(const GemmGemmParams& params, const GfxIpTriple& ip)
        {
            if (!isGfx120x(ip))
                return false;

            if (params.m == 0u || params.n == 0u || params.k == 0u || params.l == 0u)
                return false;

            if (params.dataType != MLSS_FLOAT16)
                return false;

            return true;
        }

        // Choose shader variant based on transposition flags.
        // Within the chosen tile config, transposition is the secondary axis.
        // Tile selection follows dxcp::ChooseGemmGemmShader: Wave0 for small L,
        // Wave1 for medium, Wave2 for large. Default to Wave0 (smallest safe choice).
        HipGemmGemmShader chooseShader(const GemmGemmParams& params)
        {
            const std::uint32_t base = (params.l >= 128u) ? 8u  // Wave2
                                     : (params.l >= 64u)  ? 4u  // Wave1
                                                           : 0u; // Wave0

            // Trans suffix: 0=NN, 1=TB, 2=TB0, 3=TB1
            const std::uint32_t transSuffix = params.transb0 && params.transb1 ? 1u
                                            : params.transb0                   ? 2u
                                            : params.transb1                   ? 3u
                                                                               : 0u;

            return static_cast<HipGemmGemmShader>(base + transSuffix);
        }

        const std::array<std::uint32_t, 8u>& getShaderConstants(HipGemmGemmShader shader)
        {
            switch (shader)
            {
                case HipGemmGemmShader::Shader16x64x64x32WMMA:
                case HipGemmGemmShader::Shader16x64x64x32TBWMMA:
                case HipGemmGemmShader::Shader16x64x64x32TB0WMMA:
                case HipGemmGemmShader::Shader16x64x64x32TB1WMMA:
                    return GemmGemm_16x64x64x32_CONSTANTS;

                case HipGemmGemmShader::Shader16x64x64x64WMMA:
                case HipGemmGemmShader::Shader16x64x64x64TBWMMA:
                case HipGemmGemmShader::Shader16x64x64x64TB0WMMA:
                case HipGemmGemmShader::Shader16x64x64x64TB1WMMA:
                    return GemmGemm_16x64x64x64_CONSTANTS;

                default:
                    return GemmGemm_32x64x64x128_CONSTANTS;
            }
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Cu) return {0x0Cu, 0x00u, 0x00u};
            return IP_GFX_UNKNOWN;
        }

        template <std::size_t N, std::size_t NumArgs>
        Binaries makeKernelBinaries(const std::array<std::uint8_t, N>& data,
                                    const std::array<std::uint32_t, 8u>& constants,
                                    const GemmGemmParams& params,
                                    const GfxIpTriple& gfxip,
                                    const std::array<MLSSarg, NumArgs>& args)
        {
            // Grid: ceil(M / macroTileM) × ceil(N / macroTileN) × batch
            const std::uint32_t blockCountX = integer_divide_ceil(params.m, constants[1u]);
            const std::uint32_t blockCountY = integer_divide_ceil(params.n, constants[2u]);
            const std::uint32_t threadX = constants[5u];

            MLSSdim3 grid{blockCountX, blockCountY, params.batch};
            MLSSdim3 blocks{threadX, 1u, 1u};

            Binaries binaries;

            auto relocDescriptor = make_shader_descriptor(
                std::span<const std::uint8_t>(data), "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto relocBlob = make_binary_blob(relocDescriptor);
            if (relocBlob)
            {
                *relocBlob = args;
                relocBlob->m_constants.assign(constants.begin(), constants.end());
                relocBlob->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*relocBlob));
            }

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
            nonRelocBlob = args;
            nonRelocBlob.m_constants.assign(constants.begin(), constants.end());
            nonRelocBlob.setGridBlocks(grid, blocks);
            binaries.addBlob(std::move(nonRelocBlob));

            return binaries;
        }

        std::expected<Binaries, std::error_code> buildBinariesFp16(
            const GemmGemmParams& params,
            const GfxIpTriple& gfxip,
            HipGemmGemmShader shader)
        {
            const auto& constants = getShaderConstants(shader);
            const bool isNavi44 = (gfxip.minor == 0x00u);

            if (isNavi44)
            {
                switch (shader)
                {
                    case HipGemmGemmShader::Shader16x64x64x32WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x32TBWMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Transb_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x32TB0WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Transb0_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x32TB1WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Transb1_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64TBWMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Transb_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64TB0WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Transb0_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64TB1WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Transb1_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128TBWMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Transb_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128TB0WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Transb0_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128TB1WMMA:
                        return makeKernelBinaries(fp16_1200::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Transb1_Gfx1200, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                    default:
                        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
                }
            }

            // gfx1201 (Navi48)
            switch (shader)
            {
                case HipGemmGemmShader::Shader16x64x64x32WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x32TBWMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Transb_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x32TB0WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Transb0_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x32TB1WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave0_Bs32_Lper32_Transb1_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64TBWMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Transb_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64TB0WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Transb0_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64TB1WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave1_Bs32_Lper64_Transb1_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128TBWMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Transb_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128TB0WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Transb0_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128TB1WMMA:
                    return makeKernelBinaries(fp16_1201::Gemm_Gemm_Fp16_Wave2_Bs64_Lper128_Transb1_Gfx1201, constants, params, gfxip, hip_gemm_gemm_fp16_ARGS_CONSTANTS);
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
            }
        }

        std::expected<Binaries, std::error_code> buildBinariesW8A16(
            const GemmGemmParams& params,
            const GfxIpTriple& gfxip,
            HipGemmGemmShader shader)
        {
            const auto& constants = getShaderConstants(shader);
            const bool isNavi44 = (gfxip.minor == 0x00u);

            if (isNavi44)
            {
                switch (shader)
                {
                    case HipGemmGemmShader::Shader16x64x64x32WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x32TBWMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Transb_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x32TB0WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Transb0_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x32TB1WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Transb1_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64TBWMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Transb_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64TB0WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Transb0_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader16x64x64x64TB1WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Transb1_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128TBWMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Transb_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128TB0WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Transb0_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    case HipGemmGemmShader::Shader32x64x64x128TB1WMMA:
                        return makeKernelBinaries(w8a16_1200::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Transb1_Gfx1200, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                    default:
                        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
                }
            }

            // gfx1201 (Navi48)
            switch (shader)
            {
                case HipGemmGemmShader::Shader16x64x64x32WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x32TBWMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Transb_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x32TB0WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Transb0_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x32TB1WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave0_Bs32_Lper32_Transb1_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64TBWMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Transb_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64TB0WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Transb0_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader16x64x64x64TB1WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave1_Bs32_Lper64_Transb1_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128TBWMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Transb_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128TB0WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Transb0_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                case HipGemmGemmShader::Shader32x64x64x128TB1WMMA:
                    return makeKernelBinaries(w8a16_1201::Gemm_Gemm_W8A16_Pertensor_Wave2_Bs64_Lper128_Transb1_Gfx1201, constants, params, gfxip, hip_gemm_gemm_w8a16_ARGS_CONSTANTS);
                default:
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
            }
        }

    } // anonymous namespace

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;
        const auto params = extractParams(attr);
        return isGemmGemmSupported(params, ip);
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;

        if (!isShadersAvailable(ip, attr, nullptr))
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));

        const auto params = extractParams(attr);
        const auto shader = chooseShader(params);
        const auto dtype  = chooseDtype(params);

        if (dtype == GemmGemmDtype::W8A16PerTensor)
            return buildBinariesW8A16(params, ip, shader);

        return buildBinariesFp16(params, ip, shader);
    }

} // namespace mlss::gemm_gemm::mxn::hip
