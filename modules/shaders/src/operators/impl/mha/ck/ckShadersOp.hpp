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

    // MHA Operator implementation
    class CKMha final : public BackendBase<CKMha, mlss::op::OperatorMHA>
    {
    private:

        using base = BackendBase<CKMha, mlss::op::OperatorMHA>;

        friend class BackendBase<CKMha, mlss::op::OperatorMHA>;

    public:

        // Default constructor
        CKMha() = default;

        // Constructor
        CKMha(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        // Destructor
        virtual ~CKMha() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blobs
        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::mha::ck
