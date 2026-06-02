/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "gemmMxN.hpp"
#include "shaders/src/operators/impl/gemm/utils.hpp"
#include "hip/hipGemmMxN.hpp"
#include "ck/ckGemmMxN.hpp"

template class mlss::BackendBase<mlss::gemm::mxn::hip::HipGemmMxN, mlss::gemm::mxn::GemmMxN>;
template class mlss::BackendBase<mlss::gemm::mxn::ck::CKGemmMxN, mlss::gemm::mxn::GemmMxN>;

using namespace mlss::gemm::utils;
using mlss::op::utils::MetaCmdCaps;

namespace mlss::gemm::mxn
{

    GemmMxN::GemmMxN(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "GemmMxN";
    }

    std::string GemmMxN::getCaseName()
    {
        return "AMDMLSS::GemmMxN";
    }

    std::expected<Binaries, std::error_code> GemmMxN::getBinaries() const
    {
        if (auto* hipEntry = BackendRegistry<GemmMxN>::get<hip::HipGemmMxN>(); hipEntry != nullptr)
        {
            auto result = hipEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = hipEntry->name;
                return result;
            }
        }

        if (auto* ckEntry = BackendRegistry<GemmMxN>::get<ck::CKGemmMxN>(); ckEntry != nullptr)
        {
            auto result = ckEntry->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = ckEntry->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    uint32_t GemmMxN::getCapsImpl(const std::vector<Attribute>& attributes,
                                  GfxIpTriple gfxip,
                                  const void* context)
    {
        GenericGemmParams params{};
        MetaCmdCaps caps{.values = 0x00000000u};

        if (context != nullptr)
        {
            params = *static_cast<const GenericGemmParams*>(context);
        }
        else if (!attributes.empty())
        {
            params = buildGemmParams(attributes);
        }

        if (auto* hipEntry = BackendRegistry<GemmMxN>::get<hip::HipGemmMxN>(); hipEntry != nullptr)
        {
            caps.values = hipEntry->getCaps(attributes, gfxip, &params);
        }

        if (caps.values == 0x00000000u)
        {
            if (auto* ckEntry = BackendRegistry<GemmMxN>::get<ck::CKGemmMxN>(); ckEntry != nullptr)
            {
                caps.values = ckEntry->getCaps(attributes, gfxip, &params);
            }
        }

        return caps.values;
    }

} // namespace mlss::gemm::mxn
