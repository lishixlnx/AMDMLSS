/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/gemm_gemm.hpp"
#include "impl/gemm_gemm/mxn/gemmGemmMxN.hpp"

template class mlss::CaseBase<mlss::gemm_gemm::mxn::GemmGemmMxN, mlss::op::OperatorGemmGemm>;

namespace mlss::op
{

    OperatorGemmGemm::OperatorGemmGemm(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "GEMMGEMM";
    }

    std::string OperatorGemmGemm::getOperatorName()
    {
        return "AMDMLSS::OperatorGemmGemm";
    }

    std::expected<Binaries, std::error_code> OperatorGemmGemm::getBinaries() const
    {
        auto* gemmGemm = CaseRegistry<OperatorGemmGemm>::get<mlss::gemm_gemm::mxn::GemmGemmMxN>();
        if (gemmGemm != nullptr)
        {
            auto result = gemmGemm->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = gemmGemm->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    bool OperatorGemmGemm::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto* gemmGemm = CaseRegistry<OperatorGemmGemm>::get<mlss::gemm_gemm::mxn::GemmGemmMxN>();
        if (gemmGemm != nullptr && gemmGemm->getCaps(attributes, gfxip, nullptr) != 0x00000000u)
        {
            return true;
        }

        return false;
    }

} // namespace mlss::op
