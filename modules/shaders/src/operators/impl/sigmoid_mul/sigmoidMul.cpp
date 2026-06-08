/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "sigmoidMul.hpp"
#include "hip/hipSigmoidMulShadersOp.hpp"

template class mlss::BackendBase<mlss::sigmoid_mul::hip::HipSigmoidMul, mlss::sigmoid_mul::SigmoidMul>;

namespace mlss::sigmoid_mul
{

    SigmoidMul::SigmoidMul(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "SigmoidMul";
    }

    std::string SigmoidMul::getCaseName()
    {
        return "AMDMLSS::SigmoidMul";
    }

    std::expected<Binaries, std::error_code> SigmoidMul::getBinaries() const
    {
        if (auto* hipEntry = BackendRegistry<SigmoidMul>::get<hip::HipSigmoidMul>(); hipEntry != nullptr)
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

    uint32_t SigmoidMul::getCapsImpl(const std::vector<Attribute>& attributes, GfxIpTriple gfxip, const void* context)
    {
        std::ignore = context;

        if (auto* hipEntry = BackendRegistry<SigmoidMul>::get<hip::HipSigmoidMul>(); hipEntry != nullptr)
        {
            return hipEntry->getCaps(attributes, gfxip, nullptr);
        }

        return 0x00000000u;
    }

} // namespace mlss::sigmoid_mul
