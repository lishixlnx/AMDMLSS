/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "shaders/shaders.hpp"
#include "core/core.hpp"

namespace mlss::mvn::mvn2 { class MVN2InstaNorm; }

namespace mlss::mvn::mvn2::hip
{

    class HipMvn2InstaNorm final : public BackendBase<HipMvn2InstaNorm, mlss::mvn::mvn2::MVN2InstaNorm>
    {
    private:

        using base = BackendBase<HipMvn2InstaNorm, mlss::mvn::mvn2::MVN2InstaNorm>;

    public:

        HipMvn2InstaNorm() = default;

        HipMvn2InstaNorm(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip);

        virtual ~HipMvn2InstaNorm() = default;

        static std::string getOperatorName();

        virtual std::expected<Binaries, std::error_code> getBinaries() const override;

        static bool getCapsImpl(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxArch);
    };

} // namespace mlss::mvn::mvn2::hip
