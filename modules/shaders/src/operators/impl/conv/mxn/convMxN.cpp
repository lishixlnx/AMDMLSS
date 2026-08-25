/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "convMxN.hpp"
#include "shaders/src/operators/impl/conv/utils.hpp"
#include "Misa/misaShadersOp.hpp"
#include "Winograd/Base/winogradBaseShadersOp.hpp"
#include "Winograd/Rage/winogradRageShadersOp.hpp"
#include "Winograd/Fury/winogradFuryShadersOp.hpp"

template class mlss::BackendBase<mlss::conv::mxn::misa::MisaConvMxN, mlss::conv::mxn::ConvMxN>;
template class mlss::BackendBase<mlss::conv::mxn::winograd::base::WinogradBaseConvMxN, mlss::conv::mxn::ConvMxN>;
template class mlss::BackendBase<mlss::conv::mxn::winograd::rage::WinogradRageConvMxN, mlss::conv::mxn::ConvMxN>;
template class mlss::BackendBase<mlss::conv::mxn::winograd::fury::WinogradFuryConvMxN, mlss::conv::mxn::ConvMxN>;

using namespace mlss::conv::utils;
using mlss::op::utils::MetaCmdCaps;

namespace mlss::conv::mxn
{

    namespace
    {
        // Winograd Fury non-reloc ELFs expose _amdgpu_cs_main, not the offline-linked
        // "main" entry that MIGraphX and hipModuleLoadData consumers require.
        // Prefer Winograd Base for fp16 so mlssGetBinaries returns loadable bins.
        [[nodiscard]] bool preferWinogradBaseForFp16(const GenericConvParams& params)
        {
            return params.dataType == DataTypeFlags::FLOAT16;
        }
    }

    ConvMxN::ConvMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "ConvMxN";
    }

    std::string ConvMxN::getCaseName()
    {
        return "AMDMLSS::ConvMxN";
    }

    std::expected<Binaries, std::error_code> ConvMxN::getBinaries() const
    {
        const GenericConvParams params = buildConvParams(m_attributes);
        const bool preferBase          = preferWinogradBaseForFp16(params);

        if (auto* misaEntry = BackendRegistry<ConvMxN>::get<misa::MisaConvMxN>(); misaEntry != nullptr)
        {
            auto result = misaEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = misaEntry->name;
                return result;
            }
        }

        if (auto* rageEntry = BackendRegistry<ConvMxN>::get<winograd::rage::WinogradRageConvMxN>(); rageEntry != nullptr)
        {
            auto result = rageEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = rageEntry->name;
                return result;
            }
        }

        auto* baseEntry = BackendRegistry<ConvMxN>::get<winograd::base::WinogradBaseConvMxN>();
        auto* furyEntry = BackendRegistry<ConvMxN>::get<winograd::fury::WinogradFuryConvMxN>();

        if (preferBase)
        {
            if (baseEntry != nullptr)
            {
                auto result = baseEntry->getBinaries(m_attributes, m_gfxIpTriple);
                if (result.has_value())
                {
                    m_implName = baseEntry->name;
                    return result;
                }
            }

            if (furyEntry != nullptr)
            {
                auto result = furyEntry->getBinaries(m_attributes, m_gfxIpTriple);
                if (result.has_value())
                {
                    m_implName = furyEntry->name;
                    return result;
                }
            }
        }
        else
        {
            if (furyEntry != nullptr)
            {
                auto result = furyEntry->getBinaries(m_attributes, m_gfxIpTriple);
                if (result.has_value())
                {
                    m_implName = furyEntry->name;
                    return result;
                }
            }

            if (baseEntry != nullptr)
            {
                auto result = baseEntry->getBinaries(m_attributes, m_gfxIpTriple);
                if (result.has_value())
                {
                    m_implName = baseEntry->name;
                    return result;
                }
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    uint32_t ConvMxN::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        GenericConvParams params{};
        MetaCmdCaps caps{.values = 0x00000000u};

        if (context != nullptr)
        {
            params = *static_cast<const GenericConvParams*>(context);
        }
        else if (!attributes.empty())
        {
            params = buildConvParams(attributes);
        }
        if (params.dilationX > 1 || params.dilationY > 1)
        {
            return caps.values;
        }

        if (auto* misaEntry = BackendRegistry<ConvMxN>::get<misa::MisaConvMxN>(); misaEntry != nullptr)
        {
            caps.values = misaEntry->getCaps(attributes, gfxip, &params);
        }

        if (auto* rageEntry = BackendRegistry<ConvMxN>::get<winograd::rage::WinogradRageConvMxN>(); rageEntry != nullptr)
        {
            caps.values = rageEntry->getCaps(attributes, gfxip, &params);
        }

        auto* baseEntry = BackendRegistry<ConvMxN>::get<winograd::base::WinogradBaseConvMxN>();
        auto* furyEntry = BackendRegistry<ConvMxN>::get<winograd::fury::WinogradFuryConvMxN>();

        if (preferWinogradBaseForFp16(params))
        {
            if (caps.values == 0x00000000u && baseEntry != nullptr)
            {
                caps.values = baseEntry->getCaps(attributes, gfxip, &params);
            }

            if (caps.values == 0x00000000u && furyEntry != nullptr)
            {
                caps.values = furyEntry->getCaps(attributes, gfxip, &params);
            }
        }
        else
        {
            if (caps.values == 0x00000000u && furyEntry != nullptr)
            {
                caps.values = furyEntry->getCaps(attributes, gfxip, &params);
            }

            if (caps.values == 0x00000000u && baseEntry != nullptr)
            {
                caps.values = baseEntry->getCaps(attributes, gfxip, &params);
            }
        }

        return caps.values;
    }

} // namespace mlss::conv::mxn
