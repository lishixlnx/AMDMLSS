#include "core/captions.hpp"

namespace mlss
{

    //=====================================================================================================================
    bool isGfx110x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1100) || (gfx == GfxArchitectureFlags::Gfx1101) ||
            (gfx == GfxArchitectureFlags::Gfx1102) || (gfx == GfxArchitectureFlags::Gfx1103);
    }

    //=====================================================================================================================
    bool isGfx115x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1150) || (gfx == GfxArchitectureFlags::Gfx1151) ||
            (gfx == GfxArchitectureFlags::Gfx1152) || (gfx == GfxArchitectureFlags::Gfx1153) ||
            (gfx == GfxArchitectureFlags::Gfx1154);
    }

    //=====================================================================================================================
    bool isGfx117x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1170) || (gfx == GfxArchitectureFlags::Gfx1171);
    }

    //=====================================================================================================================
    bool isGfx120x(const GfxArchitectureFlags& gfx)
    {
        return (gfx == GfxArchitectureFlags::Gfx1200) || (gfx == GfxArchitectureFlags::Gfx1201);
    }
    

    //=====================================================================================================================
    bool isGfx11(const GfxArchitectureFlags& gfx)
    {
        return isGfx110x(gfx) || isGfx115x(gfx) || isGfx117x(gfx);
    }

    //=====================================================================================================================
    bool isGfx12(const GfxArchitectureFlags& gfx)
    {
        return isGfx120x(gfx);
    }

    //=====================================================================================================================
    bool isGfx13(const GfxArchitectureFlags& gfx)
    {
        // GFX13 not yet supported
        return false;
    }

    //=====================================================================================================================
    bool isGfx11Plus(const GfxArchitectureFlags& gfx)
    {
        return isGfx11(gfx) || isGfx12Plus(gfx);
    }

    //=====================================================================================================================
    bool isGfx12Plus(const GfxArchitectureFlags& gfx)
    {
        return isGfx12(gfx) || isGfx13(gfx);
    }

   
} // mlss
