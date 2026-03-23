/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include "core/core.hpp"

namespace mlss::shaders::op
{

    class OperatorGEMM : public OperatorBase<OperatorGEMM>
    {
    private:

    using base = OperatorBase<OperatorGEMM>;
    
    // Friend declaration to allow base class access to private members
    friend class OperatorBase<OperatorGEMM>;

    public:    
        // Default constructor
        OperatorGEMM() = default;

        // Constructor
        OperatorGEMM(const std::vector<Attribute>& attributes, GfxArchitectureFlags gfxip);

        // Destructor
        virtual ~OperatorGEMM() = default;

        // Static method to get the type name for registration
        static std::string getOperatorName();

        // Override the pure virtual method to get the binary blob
        virtual std::expected<blob, std::error_code> getBlob() const override;


    private:

        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes);
    }; 
    
} // namespace mlss::shaders::op
