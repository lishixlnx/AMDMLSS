/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

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
            if (spatialSize == 0u || (spatialSize % kMvn2SpatialAlignment) != 0u)
                return false;

            if (!params.hasScale || !params.hasBias)
                return false;

            if (params.activation != MLSS_ACTIVATION_IDENTITY)
                return false;

            return true;
        }

        // Map runtime target IP to the GFX target the archived ELF was
        // compiled for. Mirrors the helper used by conv/mha/gqa backends.
        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x00u) return {0x0Bu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Bu && gfxip.minor == 0x05u) return {0x0Bu, 0x05u, 0x00u};
            if (gfxip.major == 0x0Cu)                         return {0x0Cu, 0x00u, 0x00u};
            return IP_GFX_UNKNOWN;
        }

        template <typename ShaderT, std::size_t NumArgs>
        void addKernelBlob(Binaries& binaries, const ShaderT& shader,
                           const MLSSdim3& grid, const MLSSdim3& blocks,
                           const GfxIpTriple& gfxip,
                           const std::array<MLSSarg, NumArgs>& args)
        {
            // We deliberately discard the descriptive variable name carried by
            // the archived static shader (e.g. "instaNormSplitPart1_hip_..._wave64_coba")
            // and let `make_binary_blob` extract the real ELF symbol name. The
            // runtime loaders (HIP, OpenCL) need the actual kernel symbol to
            // resolve dispatches.
            std::span<const std::uint8_t> binarySpan(
                reinterpret_cast<const std::uint8_t*>(shader.m_binary.data()),
                shader.m_binary.size());

            // 1) Relocatable variant: the original ELF blob from the archive.
            auto relocDescriptor = make_shader_descriptor(
                binarySpan, "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto relocBlob = make_binary_blob(relocDescriptor);
            if (relocBlob)
            {
                *relocBlob = args;
                relocBlob->m_constants.assign(HipMvn2_CONSTANTS.begin(),
                                              HipMvn2_CONSTANTS.end());
                relocBlob->setGridBlocks(grid, blocks);
                binaries.addBlob(std::move(*relocBlob));
            }

            // 2) Non-relocatable variant: link the relocatable ELF against
            //    the runtime target. Mirrors conv/mha/gqa logic.
            const auto sourceArch = sourceArchForTarget(gfxip);
            auto nonRelocResult = getNonRelocatable(relocDescriptor.m_binary, sourceArch, gfxip);
            if (!nonRelocResult.has_value())
            {
                return;
            }

            // Build the non-relocatable Blob and have it OWN the bytes —
            // `make_binary_blob`'s default behaviour stores a raw pointer
            // into the source span, so we must keep that source alive for
            // the lifetime of the returned Binaries. Transferring ownership
            // into the Blob via setOwnedBinary fixes the lifetime.
            auto nonRelocBytes = std::move(nonRelocResult).value();
            const std::span<const std::uint8_t> nonRelocSpan(nonRelocBytes.data(), nonRelocBytes.size());

            std::string nonRelocName;
            try
            {
                nonRelocName = getKernelName(nonRelocSpan);
            }
            catch (const std::runtime_error&) {}

            Binaries::Blob nonRelocBlob{
                nonRelocBytes.data(),
                nonRelocBytes.size(),
                static_cast<std::uint32_t>(BinaryTypeFlags::ELF),
                0u,
                nonRelocName};
            nonRelocBlob.setOwnedBinary(std::move(nonRelocBytes));
            nonRelocBlob = args;
            nonRelocBlob.m_constants.assign(HipMvn2_CONSTANTS.begin(),
                                            HipMvn2_CONSTANTS.end());
            nonRelocBlob.setGridBlocks(grid, blocks);
            binaries.addBlob(std::move(nonRelocBlob));
        }

        template <typename K1, typename K2, typename K3>
        Binaries buildInstaNormBinaries(const MVN2Params& params, const GfxIpTriple& gfxip,
                                        const K1& kernel1, const K2& kernel2, const K3& kernel3)
        {
            const std::uint32_t spatialSize = params.h * params.w;

            // Grid layout matches the reference dxcp dispatcher
            // (DdiMetaCmdMvn2InstaNorm::ExecuteMvn2):
            //   * kernel#1: hw / kernel1.macroTileSize blocks per channel.
            //   * kernel#2: one block per channel, no spatial division.
            //   * kernel#3: hw / (kernel3.threadX * elementsPerThread)
            //               blocks per channel. dxcp hardcodes the divisor
            //               to 32 * 4 = 128, which matches threadX = 32 and
            //               4 elements consumed per lane in the shipped
            //               binaries.
            const std::uint32_t kernel3TileSize =
                HipMvn2_CONSTANTS[0x0Au] * kMvn2Kernel3ElementsPerThread;

            MLSSdim3 grid1(spatialSize / HipMvn2_CONSTANTS[0x01u], params.c, 1);
            MLSSdim3 blocks1(HipMvn2_CONSTANTS[0x02u],
                             HipMvn2_CONSTANTS[0x03u],
                             HipMvn2_CONSTANTS[0x04u]);

            MLSSdim3 grid2(params.c, 1, 1);
            MLSSdim3 blocks2(HipMvn2_CONSTANTS[0x06u],
                             HipMvn2_CONSTANTS[0x07u],
                             HipMvn2_CONSTANTS[0x08u]);

            MLSSdim3 grid3(spatialSize / kernel3TileSize, params.c, 1);
            MLSSdim3 blocks3(HipMvn2_CONSTANTS[0x0Au],
                             HipMvn2_CONSTANTS[0x0Bu],
                             HipMvn2_CONSTANTS[0x0Cu]);

            Binaries binaries;
            addKernelBlob(binaries, kernel1, grid1, blocks1, gfxip,
                          mvn2_instaNorm_kernel1_ARGS_CONSTANTS);
            addKernelBlob(binaries, kernel2, grid2, blocks2, gfxip,
                          mvn2_instaNorm_kernel2_ARGS_CONSTANTS);
            addKernelBlob(binaries, kernel3, grid3, blocks3, gfxip,
                          mvn2_instaNorm_kernel3_ARGS_CONSTANTS);

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
            return buildInstaNormBinaries(params, ip,
                gfx1100::instaNormSplitPart1_hip_amdgcn_amd_amdhsa_gfx1100_wave64_coba,
                gfx1100::instaNormSplitPart2_hip_amdgcn_amd_amdhsa_gfx1100_wave32_coba,
                gfx1100::instaNormSplitPart3_hip_amdgcn_amd_amdhsa_gfx1100_wave32_coba);
        }

        if (isGfx115x(ip))
        {
            return buildInstaNormBinaries(params, ip,
                gfx1150::instaNormSplitPart1_hip_amdgcn_amd_amdhsa_gfx1150_wave64_coba,
                gfx1150::instaNormSplitPart2_hip_amdgcn_amd_amdhsa_gfx1150_wave32_coba,
                gfx1150::instaNormSplitPart3_hip_amdgcn_amd_amdhsa_gfx1150_wave32_coba);
        }

        if (isGfx120x(ip))
        {
            return buildInstaNormBinaries(params, ip,
                gfx1200::instaNormSplitPart1_hip_amdgcn_amd_amdhsa_gfx1200_wave64_coba,
                gfx1200::instaNormSplitPart2_hip_amdgcn_amd_amdhsa_gfx1200_wave32_coba,
                gfx1200::instaNormSplitPart3_hip_amdgcn_amd_amdhsa_gfx1200_wave32_coba);
        }

        return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedArchitecture));
    }

} // namespace mlss::mvn::mvn2::hip
