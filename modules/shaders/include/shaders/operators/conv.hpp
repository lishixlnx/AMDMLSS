/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::op
{
    class OperatorConv : public OperatorBase<OperatorConv>
    {
    private:

        using base = OperatorBase<OperatorConv>;

        // Friend declaration to allow base class access to private members
        friend class OperatorBase<OperatorConv>;

    public:

        // Default constructor
        OperatorConv() = default;

        // Constructor
        OperatorConv(const std::vector<Attribute>& attributes, GfxIpTriple gfxip);

        // Destructor
        virtual ~OperatorConv() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blobs
        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes);
    };

} // namespace mlss::op
