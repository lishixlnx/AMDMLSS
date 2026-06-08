/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/sigmoid_mul.hpp"
#include "impl/sigmoid_mul/sigmoidMul.hpp"

template class mlss::CaseBase<mlss::sigmoid_mul::SigmoidMul, mlss::op::OperatorSigmoidMul>;

namespace mlss::op
{

    OperatorSigmoidMul::OperatorSigmoidMul(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : base(attributes, gfxip)
    {
        this->m_implName = "SIGMOID_MUL";
    }

    std::string OperatorSigmoidMul::getOperatorName()
    {
        return "AMDMLSS::OperatorSigmoidMul";
    }

    std::expected<Binaries, std::error_code> OperatorSigmoidMul::getBinaries() const
    {
        auto* sigmoidMul = CaseRegistry<OperatorSigmoidMul>::get<mlss::sigmoid_mul::SigmoidMul>();
        if (sigmoidMul != nullptr)
        {
            auto result = sigmoidMul->getBinaries(m_attributes, m_gfxIpTriple);
            if (result.has_value())
            {
                m_implName = sigmoidMul->name;
                return result;
            }
        }

        return std::unexpected(std::make_error_code(std::errc::not_supported));
    }

    bool OperatorSigmoidMul::getCapsImpl(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
    {
        auto* sigmoidMul = CaseRegistry<OperatorSigmoidMul>::get<mlss::sigmoid_mul::SigmoidMul>();
        if (sigmoidMul != nullptr && sigmoidMul->getCaps(attributes, gfxip, nullptr) != 0x00000000u)
        {
            return true;
        }

        return false;
    }

} // namespace mlss::op
