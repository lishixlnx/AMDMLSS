/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"


namespace mlss::shaders::op
{
    class OperatorDepthWiseConv : public mlss::OperatorBase<OperatorDepthWiseConv>
    {
    private:

    using base = mlss::OperatorBase<OperatorDepthWiseConv>;

    public:    
        // Default constructor
        OperatorDepthWiseConv() = default;

        // Constructor
        OperatorDepthWiseConv(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip);

        // Destructor
        virtual ~OperatorDepthWiseConv() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blob
        virtual std::expected<blob, std::error_code> getBlob() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<mlss::Attribute>& attributes);
    };
 
    

    class OperatorDilatedConv : public mlss::OperatorBase<OperatorDilatedConv>
    {
    private:

    using base = mlss::OperatorBase<OperatorDilatedConv>;

    public:    
        // Default constructor
        OperatorDilatedConv() = default;

        // Constructor
        OperatorDilatedConv(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip);

        // Destructor
        virtual ~OperatorDilatedConv() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blob
        virtual std::expected<blob, std::error_code> getBlob() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<mlss::Attribute>& attributes);
    };


} // namespace mlss::shaders::op
