/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "shaders/operators/gqa.hpp"
#include "impl/gqa/ck/ckShadersOp.hpp"

namespace mlss::shaders::op
{
    using namespace mlss::shaders::gqa;
    
        //=====================================================================================================================
        // OperatorGQA implementation
        //=====================================================================================================================

        OperatorGQA::OperatorGQA(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
            : base(attributes, gfxip)
        {
        }

        std::string OperatorGQA::getOperatorName()
        {
            return "AMDMLSS::OperatorMHA1_t";
        }

        std::expected<Binaries, std::error_code> OperatorGQA::getBinaries() const
        {
            if(ck::CKGqa::getCapsImpl(m_attributes, m_gfxArch))
            {
                ck::CKGqa ckGqa(m_attributes, m_gfxArch);
                m_implName = ckGqa.getOperatorName();
                return ckGqa.getBinaries();
            }

            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }

        bool OperatorGQA::getCapsImpl(const std::vector<mlss::Attribute>& attributes, const GfxArchitectureFlags& gfxArch)
        {
            return ck::CKGqa::getCapsImpl(attributes, gfxArch);
        }

} // namespace mlss::shaders::op
