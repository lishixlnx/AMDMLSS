/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "rmsnorm.hpp"
#include "hip/hipRmsNormShadersOp.hpp"

template class mlss::BackendBase<mlss::rmsnorm::hip::HipRmsNorm, mlss::rmsnorm::RmsNorm>;

namespace mlss::rmsnorm
{

    RmsNorm::RmsNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "RmsNorm";
    }

    std::string RmsNorm::getCaseName()
    {
        return "AMDMLSS::RmsNorm";
    }

    std::expected<Binaries, std::error_code> RmsNorm::getBinaries() const
    {
        if (auto* hipEntry = BackendRegistry<RmsNorm>::get<hip::HipRmsNorm>(); hipEntry != nullptr)
        {
            auto result = hipEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = hipEntry->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    uint32_t RmsNorm::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        std::ignore = context;

        if (auto* hipEntry = BackendRegistry<RmsNorm>::get<hip::HipRmsNorm>(); hipEntry != nullptr)
        {
            return hipEntry->getCaps(attributes, gfxip, nullptr);
        }

        return 0x00000000u;
    }

} // namespace mlss::rmsnorm
