/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"
// Note: shaderCallHipMha.hpp has been removed
// #include "shaders/interface/shaderCallHipMha.hpp"

namespace mlss::shaders::op
{
        // MHA Operator implementation
    class OperatorMHA : public OperatorBase<OperatorMHA>
    {
    private:

    using base = OperatorBase<OperatorMHA>;

    // Friend declaration to allow base class access to private members
    friend class OperatorBase<OperatorMHA>;

    public:    
        // Default constructor
        OperatorMHA() = default;
        
        // Constructor
        OperatorMHA(const std::vector<Attribute>& attributes, GfxArchitectureFlags gfxip);

        // Destructor
        virtual ~OperatorMHA() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blob
        virtual std::expected<blob, std::error_code> getBlob() const override;

    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes);
        
    };

} // namespace mlss::shaders::op
