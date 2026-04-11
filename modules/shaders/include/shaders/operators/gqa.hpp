/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::op
{

    class OperatorGQA : public OperatorBase<OperatorGQA>
    {
    private:

        using base = OperatorBase<OperatorGQA>;

        // Friend declaration to allow base class access to private members
        friend class OperatorBase<OperatorGQA>;

    public:

        // Default constructor
        OperatorGQA() = default;

        // Constructor
        OperatorGQA(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        // Destructor
        virtual ~OperatorGQA() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blobs
        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);

    private:
    };

} // namespace mlss::op
