/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::rmsnorm { class RmsNorm; }

namespace mlss::rmsnorm::hip
{

    class HipRmsNorm final : public BackendBase<HipRmsNorm, mlss::rmsnorm::RmsNorm>
    {
    private:

        using base = BackendBase<HipRmsNorm, mlss::rmsnorm::RmsNorm>;

    public:

        HipRmsNorm() = default;

        HipRmsNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipRmsNorm() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::rmsnorm::hip
