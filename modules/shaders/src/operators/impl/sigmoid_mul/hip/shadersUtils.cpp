/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "shadersUtils.hpp"
#include "shadersConstants.hpp"

#include "fp16/gfx1200/shadersBin.hpp"
#include "fp16/gfx1201/shadersBin.hpp"

namespace gfx1200 = mlss::sigmoid_mul::hip::fp16::gfx1200;
namespace gfx1201 = mlss::sigmoid_mul::hip::fp16::gfx1201;

namespace mlss::sigmoid_mul::hip
{

    using mlss::isGfx120x;

    namespace
    {

        struct SigmoidMulParams
        {
            std::uint32_t n{0};
            std::uint32_t c{0};
            std::uint32_t h{0};
            std::uint32_t w{0};
            std::uint32_t dataType{0};
        };

        SigmoidMulParams extractParams(const std::vector<Attribute>& attr)
        {
            SigmoidMulParams params;
            for (const auto& attribute : attr)
            {
                if (attribute.is(MLSS_ATTR_SIGMOID_MUL_N))
                    params.n = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_SIGMOID_MUL_C))
                    params.c = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_SIGMOID_MUL_H))
                    params.h = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_SIGMOID_MUL_W))
                    params.w = attribute.value<std::uint32_t>();
                else if (attribute.is(MLSS_ATTR_SIGMOID_MUL_DATATYPE))
                    params.dataType = attribute.value<std::uint32_t>();
            }
            return params;
        }

        bool isSigmoidMulSupported(const SigmoidMulParams& params, const GfxIpTriple& ip)
        {
            if (!isGfx120x(ip))
                return false;

            if (params.n == 0u || params.c == 0u || params.h == 0u || params.w == 0u)
                return false;

            if (params.dataType != MLSS_FLOAT16)
                return false;

            return true;
        }

        GfxIpTriple sourceArchForTarget(const GfxIpTriple& gfxip)
        {
            if (gfxip.major == 0x0Cu && gfxip.minor == 0x00u) return {0x0Cu, 0x00u, 0x00u};
            if (gfxip.major == 0x0Cu)                         return {0x0Cu, 0x00u, 0x01u};
            return IP_GFX_UNKNOWN;
        }

        template <std::size_t N>
        Binaries makeKernelBinaries(const std::array<std::uint8_t, N>& data,
                                    const SigmoidMulParams& params,
                                    const GfxIpTriple& gfxip)
        {
            // Grid: ceil(N*C*H*W / blockSize) blocks in X, 1 in Y/Z.
            // blockSize = 64 (SigmoidMul_CONSTANTS[0]).
            const std::uint32_t blockSize = SigmoidMul_CONSTANTS[0u];
            const std::uint32_t totalElements = params.n * params.c * params.h * params.w;
            const std::uint32_t blockCountX = integer_divide_ceil(totalElements, blockSize);
            MLSSdim3 grid{blockCountX, 1u, 1u};
            MLSSdim3 blocks{blockSize, 1u, 1u};

            Binaries binaries;

            // 1) Relocatable variant.
            auto relocDescriptor = make_shader_descriptor(
                std::span<const std::uint8_t>(data), "", "", 0, true, ShaderTypesFlags::UNKNOWN);
            auto relocBlob = make_binary_blob(relocDescriptor);
            if (relocBlob)
            {
                *relocBlob = hip_sigmoid_mul_ARGS_CONSTANTS;
                relocBlob->m_constants.assign(SigmoidMul_CONSTANTS.begin(), SigmoidMul_CONSTANTS.end());
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
            nonRelocBlob = hip_sigmoid_mul_ARGS_CONSTANTS;
            nonRelocBlob.m_constants.assign(SigmoidMul_CONSTANTS.begin(), SigmoidMul_CONSTANTS.end());
            nonRelocBlob.setGridBlocks(grid, blocks);
            binaries.addBlob(std::move(nonRelocBlob));

            return binaries;
        }

    } // anonymous namespace

    bool isShadersAvailable(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;
        const auto params = extractParams(attr);
        return isSigmoidMulSupported(params, ip);
    }

    std::expected<Binaries, std::error_code> getShadersBlob(const GfxIpTriple& ip, const std::vector<Attribute>& attr, const void* cstmStruct)
    {
        std::ignore = cstmStruct;

        if (!isShadersAvailable(ip, attr, nullptr))
            return std::unexpected(make_error_code(MLSSErrorCode::ShaderUnsupportedConfiguration));

        const auto params = extractParams(attr);

        // gfx1200 = Navi44 (minor==0x00), gfx1201 = Navi48 (minor==0x01).
        if (ip.minor == 0x00u)
            return makeKernelBinaries(gfx1200::SigmoidMulFp16Gfx1200, params, ip);

        return makeKernelBinaries(gfx1201::SigmoidMulFp16Gfx1201, params, ip);
    }

} // namespace mlss::sigmoid_mul::hip
