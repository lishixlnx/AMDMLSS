/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "gemmGemmMxN.hpp"
#include "hip/hipGemmGemmShadersOp.hpp"

template class mlss::BackendBase<mlss::gemm_gemm::mxn::hip::HipGemmGemmMxN, mlss::gemm_gemm::mxn::GemmGemmMxN>;

namespace mlss::gemm_gemm::mxn
{

    GemmGemmMxN::GemmGemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "GemmGemm-MxN";
    }

    std::string GemmGemmMxN::getCaseName()
    {
        return "AMDMLSS::GemmGemmMxN";
    }

    std::expected<Binaries, std::error_code> GemmGemmMxN::getBinaries() const
    {
        if (auto* hipEntry = BackendRegistry<GemmGemmMxN>::get<hip::HipGemmGemmMxN>(); hipEntry != nullptr)
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

    uint32_t GemmGemmMxN::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        std::ignore = context;

        if (auto* hipEntry = BackendRegistry<GemmGemmMxN>::get<hip::HipGemmGemmMxN>(); hipEntry != nullptr)
        {
            return hipEntry->getCaps(attributes, gfxip, nullptr);
        }

        return 0x00000000u;
    }

} // namespace mlss::gemm_gemm::mxn
