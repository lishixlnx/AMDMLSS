/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "shaders/shaders.hpp"
#include "core/core.hpp"
#include <cstdint>

namespace mlss::shaders::op
{
    class OperatorMHA;
}

namespace mlss::shaders::mha::ck
{

    // MHA shader variants for CK implementation
    enum class MHAAsmShaderWmma : std::uint32_t
    {
        unknown = 0,
        // Unpacked variants (0-4)
        unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16,
        unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16,
        unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16,
        unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16,
        unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16,

        // Packed offset constant
        PackedOffset = 5,

        // Packed variants - defined as offset from unpacked
        packed_kv_cross_attention_128_64x64x48_64x48x64_forward_fp16 = unpacked_cross_attention_128_64x64x48_64x48x64_forward_fp16 + PackedOffset,
        packed_kv_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16 = unpacked_fallback_cross_attention_64_32x64x48_32x48x64_forward_fp16 + PackedOffset,
        packed_qkv_self_attention_128_64x128x80_64x80x64_forward_fp16 = unpacked_self_attention_128_64x128x80_64x80x64_forward_fp16 + PackedOffset,
        packed_qkv_self_attention_128_64x192x48_64x48x64_forward_fp16 = unpacked_self_attention_128_64x192x48_64x48x64_forward_fp16 + PackedOffset,
        packed_qkv_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16 = unpacked_fallback_self_attention_64_32x64x48_32x48x64_forward_fp16 + PackedOffset
    };

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
        
        // Override the pure virtual method to get the binary blob
        virtual std::expected<blob, std::error_code> getBlob() const override;
        
    private:
        // Static method to check capabilities
        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxArchitectureFlags& gfxArch);
    };

} // namespace mlss::shaders::mha::ck::wmma
