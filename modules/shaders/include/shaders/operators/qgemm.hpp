/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::op
{

    class OperatorQGEMM : public OperatorBase<OperatorQGEMM>
    {
    private:

        using base = OperatorBase<OperatorQGEMM>;

        // Friend declaration to allow base class access to private members
        friend class OperatorBase<OperatorQGEMM>;

    public:

        // Default constructor
        OperatorQGEMM() = default;

        // Constructor
        OperatorQGEMM(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        // Destructor
        virtual ~OperatorQGEMM() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blobs
        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes);
    };

} // namespace mlss::op
