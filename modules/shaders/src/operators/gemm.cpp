/* Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/gemm.hpp"
#include "impl/gemm/mxn/gemmMxN.hpp"

template class mlss::CaseBase<mlss::gemm::mxn::GemmMxN, mlss::op::OperatorGEMM>;

namespace mlss::op
{

    OperatorGEMM::OperatorGEMM(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
    }

    std::string OperatorGEMM::getOperatorName()
    {
        return "AMDMLSS::OperatorGEMM";
    }

    std::expected<Binaries, std::error_code> OperatorGEMM::getBinaries() const
    {
        auto* gemmMxN = CaseRegistry<OperatorGEMM>::get<mlss::gemm::mxn::GemmMxN>();
        if (gemmMxN != nullptr)
        {
            auto result = gemmMxN->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = gemmMxN->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    bool OperatorGEMM::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto* gemmMxN = CaseRegistry<OperatorGEMM>::get<mlss::gemm::mxn::GemmMxN>();
        if (gemmMxN != nullptr && gemmMxN->getCaps(attributes, gfxip, nullptr) != 0x00000000u)
        {
            return true;
        }

        return false;
    }

} // namespace mlss::op
