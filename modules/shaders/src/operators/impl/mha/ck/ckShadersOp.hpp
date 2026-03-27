/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::shaders::op
{
    class OperatorMHA;
}

namespace mlss::shaders::mha::ck
{

    // MHA Operator implementation
    class CKMha final : public OperatorBase<CKMha, OperatorRegistration::Disabled>
    {
    private:

        using base = OperatorBase<CKMha, OperatorRegistration::Disabled>;

        // Friend declaration to allow base class access to private members
        friend class OperatorBase<CKMha, OperatorRegistration::Disabled>;
        friend class mlss::shaders::op::OperatorMHA;

    public:

        // Default constructor
        CKMha() = default;

        // Constructor
        CKMha(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxip);

        // Destructor
        virtual ~CKMha() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blobs
        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxArch);
    };

} // namespace mlss::shaders::mha::ck
