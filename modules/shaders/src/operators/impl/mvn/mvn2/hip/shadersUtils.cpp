/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"

#include "gfx1100/fp16/shadersBin.hpp"
#include "gfx1150/fp16/shadersBin.hpp"
#include "gfx1200/fp16/shadersBin.hpp"

namespace gfx1100 = mlss::mvn::mvn2::hip::fp16::gfx1100;
namespace gfx1150 = mlss::mvn::mvn2::hip::fp16::gfx1150;
namespace gfx1200 = mlss::mvn::mvn2::hip::fp16::gfx1200;

namespace mlss::mvn::mvn2::hip
{

    using mlss::isGfx110x;
    using mlss::isGfx115x;
    using mlss::isGfx120x;

    namespace
    {

        constexpr std::uint32_t kSpatialAlignment = 256u;
        constexpr std::uint32_t kKernel1WaveSize = 64u;
        constexpr std::uint32_t kKernel2WaveSize = 32u;
        constexpr std::uint32_t kKernel3WaveSize = 32u;

        struct MVN2Params
        {
            std::uint32_t n{0};
            std::uint32_t c{0};
            std::uint32_t h{0};
            std::uint32_t w{0};
            std::uint32_t dataType{0};
            bool crossChannel{false};
            bool hasScale{false};
            bool hasBias{false};
            std::uint32_t activation{0};
        };

        MVN2Params extractParams(const std::vector<Attribute>& attr)
        {
            MVN2Params params;

            for (const auto& attribute : attr)
            {
                if (attribute.is(MLSS_ATTR_MVN_N))
                    params.n = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_MVN_C))
                    params.c = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_MVN_H))
                    params.h = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_MVN_W))
                    params.w = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_MVN_DATATYPE))
                    params.dataType = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_MVN_CROSSCHANNEL))
                    params.crossChannel = attribute.value<bool>();
                else if (attribute.is(MLSS_ATTR_MVN_HASSCALE))
                    params.hasScale = attribute.value<bool>();
                else if (attribute.is(MLSS_ATTR_MVN_HASBIAS))
                    params.hasBias = attribute.value<bool>();
                else if (attribute.is(MLSS_ATTR_MVN_ACTIVATION))
                    params.activation = attribute.value<std::uint32_t>();
            }

            return params;
        }

        bool isInstaNormSupported(const MVN2Params& params, const GfxIpTriple& ip)
        {
            if (!isGfx110x(ip) && !isGfx115x(ip) && !isGfx120x(ip))
                return false;

            if (params.n != 1u)
                return false;

            if (params.dataType != MLSS_FLOAT16)
                return false;

            if (params.crossChannel)
                return false;

            const std::uint32_t spatialSize = params.h * params.w;
            if (spatialSize == 0u || (spatialSize % kSpatialAlignment) != 0u)
                return false;

            if (!params.hasScale || !params.hasBias)
                return false;

            if (params.activation != MLSS_ACTIVATION_IDENTITY)
                return false;

            return true;
        }

        template <typename ShaderT>
        void addKernelBlob(Binaries& binaries, const ShaderT& shader,
                           const MLSSdim3& grid, const MLSSdim3& blocks)
        {
            auto blobPtr = make_binary_blob(shader);
            if (blobPtr)
            {
                blobPtr->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*blobPtr));
            }
        }

        template <typename K1, typename K2, typename K3>
        Binaries buildInstaNormBinaries(const MVN2Params& params,
                                        const K1& kernel1, const K2& kernel2, const K3& kernel3)
        {
            const std::uint32_t spatialSize = params.h * params.w;
            const std::uint32_t numSpatialGroups = spatialSize / kSpatialAlignment;

            MLSSdim3 grid1(params.c * numSpatialGroups, 1, 1);
            MLSSdim3 blocks1(kKernel1WaveSize, 1, 1);

            MLSSdim3 grid2(params.c, 1, 1);
            MLSSdim3 blocks2(kKernel2WaveSize, 1, 1);

            MLSSdim3 grid3(params.c * numSpatialGroups, 1, 1);
            MLSSdim3 blocks3(kKernel3WaveSize, 1, 1);

            Binaries binaries;
            addKernelBlob(binaries, kernel1, grid1, blocks1);
            addKernelBlob(binaries, kernel2, grid2, blocks2);
            addKernelBlob(binaries, kernel3, grid3, blocks3);

            return binaries;
        }

    } // anonymous namespace

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;

        const auto params = extractParams(attr);
        return isInstaNormSupported(params, ip);
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        if (!isShadersAvailable(ip, attr, cstmStruct))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));
        }

        const auto params = extractParams(attr);

        if (isGfx110x(ip))
        {
            return buildInstaNormBinaries(params,
                gfx1100::instaNormSplitPart1_hip_amdgcn_amd_amdhsa_gfx1100_wave64_coba,
                gfx1100::instaNormSplitPart2_hip_amdgcn_amd_amdhsa_gfx1100_wave32_coba,
                gfx1100::instaNormSplitPart3_hip_amdgcn_amd_amdhsa_gfx1100_wave32_coba);
        }

        if (isGfx115x(ip))
        {
            return buildInstaNormBinaries(params,
                gfx1150::instaNormSplitPart1_hip_amdgcn_amd_amdhsa_gfx1150_wave64_coba,
                gfx1150::instaNormSplitPart2_hip_amdgcn_amd_amdhsa_gfx1150_wave32_coba,
                gfx1150::instaNormSplitPart3_hip_amdgcn_amd_amdhsa_gfx1150_wave32_coba);
        }

        if (isGfx120x(ip))
        {
            return buildInstaNormBinaries(params,
                gfx1200::instaNormSplitPart1_hip_amdgcn_amd_amdhsa_gfx1200_wave64_coba,
                gfx1200::instaNormSplitPart2_hip_amdgcn_amd_amdhsa_gfx1200_wave32_coba,
                gfx1200::instaNormSplitPart3_hip_amdgcn_amd_amdhsa_gfx1200_wave32_coba);
        }

        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
    }

} // namespace mlss::mvn::mvn2::hip
