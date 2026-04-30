/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::op
{
    class OperatorMHA;
}

namespace mlss::mha::ck
{

    class CKMha final : public BackendBase<CKMha, mlss::op::OperatorMHA>
    {
    private:

        using base = BackendBase<CKMha, mlss::op::OperatorMHA>;

    public:

        CKMha() = default;

        CKMha(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~CKMha() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::mha::ck
