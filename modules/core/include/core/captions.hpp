#pragma once
#include "core/core.hpp"

namespace mlss
{

    bool isGfx110x(const GfxArchitectureFlags& gfx);
    bool isGfx115x(const GfxArchitectureFlags& gfx);
    bool isGfx117x(const GfxArchitectureFlags& gfx);
    bool isGfx120x(const GfxArchitectureFlags& gfx);
    bool isGfx13x(const GfxArchitectureFlags& gfx);

    bool isGfx11Plus(const GfxArchitectureFlags& gfx);
    bool isGfx12Plus(const GfxArchitectureFlags& gfx);

} // mlss
