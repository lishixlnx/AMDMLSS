/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"

#include "gfx1201/fp32/shadersBin.hpp"
#include "gfx1201/fp32/DecisionTree.hpp"

namespace ck_gfx1201 = mlss::gemm::mxn::ck::fp32::gfx1201;

namespace mlss::gemm::mxn::ck
{

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
            }

            return params;
        }

        bool isCKGemmSupported(const GemmParams& params, const GfxIpTriple& ip)
        {
            if (!isGfx120x(ip))
                return false;

            if (params.dataType != MLSS_FLOAT32)
                return false;

            if (params.m == 0 || params.n == 0 || params.k == 0)
                return false;

            if (params.transA || params.transB)
                return false;

            return true;
        }

        int traverseDecisionTree(float m, float n, float k)
        {
            const std::array<float, 3> features = {m, n, k};
            return mlss::predictHelper(ck_gfx1201::Navi48CKGemmFp32Tree,
                                       std::span<const float>(features));
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

    } // anonymous namespace

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;

        const auto params = extractParams(attr);
        return isCKGemmSupported(params, ip);
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        if (!isShadersAvailable(ip, attr, cstmStruct))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
        }

        const auto params = extractParams(attr);

        const int treeResult = traverseDecisionTree(
            static_cast<float>(params.m),
            static_cast<float>(params.n),
            static_cast<float>(params.k));

        std::ignore = treeResult;

        constexpr std::uint32_t kDefaultBlockSize = 256u;
        constexpr std::uint32_t kDefaultTileM = 64u;
        constexpr std::uint32_t kDefaultTileN = 64u;

        MLSSdim3 grid(divCeil(params.m, kDefaultTileM), divCeil(params.n, kDefaultTileN), params.batch);
        MLSSdim3 blocks(kDefaultBlockSize, 1, 1);

        Binaries binaries;
        addKernelBlob(binaries, ck_gfx1201::gemm_add_fp32_add_dl_nn_gfx1201, grid, blocks);

        return binaries;
    }

} // namespace mlss::gemm::mxn::ck
