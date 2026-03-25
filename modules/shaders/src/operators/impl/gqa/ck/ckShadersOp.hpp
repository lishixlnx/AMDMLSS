/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "shaders/shaders.hpp"
#include "core/core.hpp"
#include <cstdint>

namespace mlss::shaders::op
{
    class OperatorGQA;
}

namespace mlss::shaders::gqa::ck
{

    // GQA shader variants for CK implementation
    enum class GQAAsmShaderWmma : std::uint32_t
    {
        unknown = 0,
        // Unpacked double pointer variants
        unpacked_double_pointer_128_64x128x80_64x80x64,
        unpacked_double_pointer_128_64x192x48_64x48x64,
        unpacked_double_pointer_128_64x64x48_64x48x64,
        unpacked_double_pointer_64_32x64x48_32x48x64,

        // Unpacked with strides variants
        unpacked_with_strides_128_64x128x80_64x80x64,
        unpacked_with_strides_128_64x192x48_64x48x64,
        unpacked_with_strides_128_64x64x48_64x48x64,
        unpacked_with_strides_64_32x64x48_32x48x64
    };

    // GQA Operator implementation using CK library
    class CKGqa final : public OperatorBase<CKGqa, OperatorRegistration::Disabled>
    {
    private:
        using base = OperatorBase<CKGqa, OperatorRegistration::Disabled>;
        
        // Friend declaration to allow base class access to private members
        friend class OperatorBase<CKGqa, OperatorRegistration::Disabled>;
        friend class mlss::shaders::op::OperatorGQA;
        
    public:    
        // Default constructor
        CKGqa() = default;
        
        // Constructor
        CKGqa(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxip);
        
        // Destructor
        virtual ~CKGqa() = default;
        
        // Static method to get the type name for registration
        static std::string getOperatorName();
        
        // Override the pure virtual method to get the binary blobs
        virtual std::expected<Binaries, std::error_code> getBinaries() const override;
        
    private:
        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxArch);
    };

} // namespace mlss::shaders::gqa::ck

