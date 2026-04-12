#pragma once

#include "core/core.hpp"

namespace mlss::op::utils
{
    union MetaCmdCaps
    {
        struct
        {
            uint32 support     :  1; // Basic support, there may be performance and precision issues
            uint32 fullSupport :  1; // Full support
            uint32 reserved    : 30;
        };
    
        uint32 values;
    };
    
} // namespace mlss::op::utils