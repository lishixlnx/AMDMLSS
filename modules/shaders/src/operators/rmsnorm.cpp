/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/rmsnorm.hpp"
#include "impl/rmsnorm/rmsnorm.hpp"

template class mlss::CaseBase<mlss::rmsnorm::RmsNorm, mlss::op::OperatorRmsNorm>;

namespace mlss::op
{

    OperatorRmsNorm::OperatorRmsNorm(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "RMSNORM";
    }

    std::string OperatorRmsNorm::getOperatorName()
    {
        return "AMDMLSS::OperatorRmsNorm";
    }

    std::expected<Binaries, std::error_code> OperatorRmsNorm::getBinaries() const
    {
        auto* rmsNorm = CaseRegistry<OperatorRmsNorm>::get<mlss::rmsnorm::RmsNorm>();
        if (rmsNorm != nullptr)
        {
            auto result = rmsNorm->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = rmsNorm->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    bool OperatorRmsNorm::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto* rmsNorm = CaseRegistry<OperatorRmsNorm>::get<mlss::rmsnorm::RmsNorm>();
        if (rmsNorm != nullptr && rmsNorm->getCaps(attributes, gfxip, nullptr) != 0x00000000u)
        {
            return true;
        }

        return false;
    }

} // namespace mlss::op
