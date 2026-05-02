/* Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/mvn.hpp"
#include "impl/mvn/mvn2/mvn2InstaNorm.hpp"

template class mlss::CaseBase<mlss::mvn::mvn2::MVN2InstaNorm, mlss::op::OperatorMVN>;

namespace mlss::op
{

    OperatorMVN::OperatorMVN(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "MVN";
    }

    std::string OperatorMVN::getOperatorName()
    {
        return "AMDMLSS::OperatorMVN";
    }

    std::expected<Binaries, std::error_code> OperatorMVN::getBinaries() const
    {
        auto* mvn2 = CaseRegistry<OperatorMVN>::get<mlss::mvn::mvn2::MVN2InstaNorm>();
        if (mvn2 != nullptr)
        {
            auto result = mvn2->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = mvn2->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    bool OperatorMVN::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto* mvn2 = CaseRegistry<OperatorMVN>::get<mlss::mvn::mvn2::MVN2InstaNorm>();
        if (mvn2 != nullptr && mvn2->getCaps(attributes, gfxip, nullptr) != 0x00000000u)
        {
            return true;
        }

        return false;
    }

} // namespace mlss::op
