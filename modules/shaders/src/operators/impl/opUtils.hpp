#pragma once

#include "core/core.hpp"

namespace mlss::op::utils
{
    union MetaCmdCaps
    {
        struct
        {
            std::uint32_t support     :  1;
            std::uint32_t fullSupport :  1;
            std::uint32_t reserved    : 30;
        };

        std::uint32_t values;
    };
    
} // namespace mlss::op::utils