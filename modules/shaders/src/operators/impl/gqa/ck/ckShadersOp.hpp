/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once
#include "shaders/shaders.hpp"
#include "core/core.hpp"
#include <cstdint>

namespace mlss::op
{
    class OperatorGQA;
}

namespace mlss::gqa::ck
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

    class CKGqa final : public BackendBase<CKGqa, mlss::op::OperatorGQA>
    {
    private:

        using base = BackendBase<CKGqa, mlss::op::OperatorGQA>;

    public:

        CKGqa() = default;

        CKGqa(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~CKGqa() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::gqa::ck
