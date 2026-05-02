/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"

#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1201/fp16/shadersBin.hpp"

namespace gfx1100 = mlss::gemm::mxn::hip::fp16::gfx1100;
namespace gfx1150 = mlss::gemm::mxn::hip::fp16::gfx1150;
namespace gfx1201 = mlss::gemm::mxn::hip::fp16::gfx1201;
namespace gfx1201_noact = mlss::gemm::mxn::hip::fp16::gfx1201::noActivations;

namespace mlss::gemm::mxn::hip
{

    using mlss::isGfx110x;
    using mlss::isGfx115x;
    using mlss::isGfx120x;

    namespace
    {

        struct GemmParams
        {
            std::uint32_t m{0};
            std::uint32_t n{0};
            std::uint32_t k{0};
            std::uint32_t batch{1};
            bool transA{false};
            bool transB{false};
            std::uint32_t dataType{0};
            std::uint32_t activation{0};
        };

        GemmParams extractParams(const std::vector<Attribute>& attr)
        {
            GemmParams params;

            for (const auto& attribute : attr)
            {
                if (attribute.is(MLSS_ATTR_GEMM_M))
                    params.m = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMM_N))
                    params.n = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMM_K))
                    params.k = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMM_BATCH))
                    params.batch = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMM_TRANSA))
                    params.transA = attribute.value<bool>();
                else if (attribute.is(MLSS_ATTR_GEMM_TRANSB))
                    params.transB = attribute.value<bool>();
                else if (attribute.is(MLSS_ATTR_GEMM_DATATYPE))
                    params.dataType = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_GEMM_ACTIVATION))
                    params.activation = attribute.value<std::uint32_t>();
            }

            return params;
        }

        bool isHipGemmSupported(const GemmParams& params, const GfxIpTriple& ip)
        {
            if (!isGfx110x(ip) && !isGfx115x(ip) && !isGfx120x(ip))
                return false;

            if (params.dataType != MLSS_FLOAT16)
                return false;

            if (params.m == 0 || params.n == 0 || params.k == 0)
                return false;

            if (params.transA)
                return false;

            return true;
        }

        constexpr std::uint32_t divCeil(std::uint32_t a, std::uint32_t b)
        {
            return (a + b - 1) / b;
        }

        template <std::size_t N>
        void addKernelBlob(Binaries& binaries, const std::array<std::uint8_t, N>& data,
                           const MLSSdim3& grid, const MLSSdim3& blocks)
        {
            auto desc = make_shader_descriptor(
                std::span<const std::uint8_t>(data), "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto blob = make_binary_blob(desc);
            if (blob)
            {
                blob->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*blob));
            }
        }

        template <std::size_t N>
        Binaries buildSingleKernelBinaries(const GemmParams& params,
                                           const std::array<std::uint8_t, N>& kernel,
                                           std::uint32_t tileM, std::uint32_t tileN,
                                           std::uint32_t blockSize)
        {
            MLSSdim3 grid(divCeil(params.m, tileM), divCeil(params.n, tileN), params.batch);
            MLSSdim3 blocks(blockSize, 1, 1);

            Binaries binaries;
            addKernelBlob(binaries, kernel, grid, blocks);
            return binaries;
        }

        template <typename GetNN, typename GetNT>
        std::expected<Binaries, std::error_code> selectAndBuild(
            const GemmParams& params, GetNN getNNKernel, GetNT getNTKernel)
        {
            if (params.transB)
            {
                return getNTKernel(params);
            }
            return getNNKernel(params);
        }

        std::expected<Binaries, std::error_code> buildGfx1100(const GemmParams& params)
        {
            return selectAndBuild(params,
                [](const GemmParams& p) -> std::expected<Binaries, std::error_code>
                {
                    if (p.m >= 64 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1100::gemm_add_fp16_64x128x32_add_nn_gfx1100, 64, 128, 256);
                    if (p.m >= 16 && p.n >= 256)
                        return buildSingleKernelBinaries(p, gfx1100::gemm_add_fp16_16x256x16_add_1_2_nn_gfx1100, 16, 256, 256);
                    if (p.m >= 16 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1100::gemm_add_fp16_16x128x16_add_nn_gfx1100, 16, 128, 256);
                    return buildSingleKernelBinaries(p, gfx1100::gemm_add_fp16_16x16x16_add_nn_gfx1100, 16, 16, 256);
                },
                [](const GemmParams& p) -> std::expected<Binaries, std::error_code>
                {
                    return buildSingleKernelBinaries(p, gfx1100::gemm_add_fp16_64x128x32_add_nt_gfx1100, 64, 128, 256);
                });
        }

        std::expected<Binaries, std::error_code> buildGfx1150(const GemmParams& params)
        {
            return selectAndBuild(params,
                [](const GemmParams& p) -> std::expected<Binaries, std::error_code>
                {
                    if (p.m >= 64 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1150::gemm_add_fp16_64x128x32_add_nn_gfx1150, 64, 128, 256);
                    if (p.m >= 16 && p.n >= 256)
                        return buildSingleKernelBinaries(p, gfx1150::gemm_add_fp16_16x256x16_add_1_2_nn_gfx1150, 16, 256, 256);
                    if (p.m >= 16 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1150::gemm_add_fp16_16x128x16_add_nn_gfx1150, 16, 128, 256);
                    return buildSingleKernelBinaries(p, gfx1150::gemm_add_fp16_16x16x16_add_nn_gfx1150, 16, 16, 256);
                },
                [](const GemmParams& p) -> std::expected<Binaries, std::error_code>
                {
                    return buildSingleKernelBinaries(p, gfx1150::gemm_add_fp16_64x128x32_add_nt_gfx1150, 64, 128, 256);
                });
        }

        std::expected<Binaries, std::error_code> buildGfx1201(const GemmParams& params)
        {
            const bool useNoActivation = (params.activation == MLSS_ACTIVATION_IDENTITY);

            return selectAndBuild(params,
                [useNoActivation](const GemmParams& p) -> std::expected<Binaries, std::error_code>
                {
                    if (useNoActivation)
                    {
                        if (p.m >= 128 && p.n >= 128)
                            return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_128x128x16_add_nn_gfx1201, 128, 128, 256);
                        if (p.m >= 128 && p.n >= 64)
                            return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_128x64x32_add_tr_nn_gfx1201, 128, 64, 256);
                        if (p.m >= 64 && p.n >= 128)
                            return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_64x128x32_add_tr_nn_gfx1201, 64, 128, 256);
                        if (p.m >= 256 && p.n >= 16)
                            return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_256x16x16_add_2_1_nn_gfx1201, 256, 16, 256);
                        if (p.m >= 16 && p.n >= 256)
                            return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_16x256x16_add_1_2_nn_gfx1201, 16, 256, 256);
                        if (p.m >= 16 && p.n >= 128)
                            return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_16x128x16_add_nn_gfx1201, 16, 128, 256);
                        return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_16x16x16_add_native_nn_gfx1201, 16, 16, 256);
                    }

                    if (p.m >= 128 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_128x128x16_add_nn_gfx1201, 128, 128, 256);
                    if (p.m >= 128 && p.n >= 64)
                        return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_128x64x32_add_nn_gfx1201, 128, 64, 256);
                    if (p.m >= 64 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_64x128x32_add_nn_gfx1201, 64, 128, 256);
                    if (p.m >= 256 && p.n >= 16)
                        return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_256x16x16_add_nn_gfx1201, 256, 16, 256);
                    if (p.m >= 16 && p.n >= 256)
                        return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_16x256x16_add_nn_gfx1201, 16, 256, 256);
                    if (p.m >= 16 && p.n >= 128)
                        return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_16x128x16_add_nn_gfx1201, 16, 128, 256);
                    return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_16x16x16_add_nn_gfx1201, 16, 16, 256);
                },
                [useNoActivation](const GemmParams& p) -> std::expected<Binaries, std::error_code>
                {
                    if (useNoActivation)
                        return buildSingleKernelBinaries(p, gfx1201_noact::gemm_add_fp16_64x128x32_add_2_4_nt_gfx1201, 64, 128, 256);
                    return buildSingleKernelBinaries(p, gfx1201::gemm_add_fp16_64x128x32_add_nt_gfx1201, 64, 128, 256);
                });
        }

    } // anonymous namespace

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;

        const auto params = extractParams(attr);
        return isHipGemmSupported(params, ip);
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        if (!isShadersAvailable(ip, attr, cstmStruct))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
        }

        const auto params = extractParams(attr);

        if (isGfx110x(ip))
            return buildGfx1100(params);

        if (isGfx115x(ip))
            return buildGfx1150(params);

        if (isGfx120x(ip))
            return buildGfx1201(params);

        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
    }

} // namespace mlss::gemm::mxn::hip
