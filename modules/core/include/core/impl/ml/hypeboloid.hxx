#pragma once

namespace mlss
{

    // This struct holds constants that define some 3D hyperboloid of the form:
    //     (c - cOffset) * (k - kOffset) * (d - xyOffset^2) - dist^3 = 0
    // If the left side is >= 0 we'll say that the (c, k, d) point is inside the hyperboloid surface. For a properly tuned
    // set of constants, any point inside the surface should run faster on our metacommand path than the MS shaders.
    // All constants should fit within 16 bits in the value ranges we care about which halves the space needed for tuning.
    union HyperConsts
    {
        struct
        {
            std::uint16_t cOffset;
            std::uint16_t kOffset;
            std::uint16_t xyOffset; // Really the sqrt of the combined d offset. We take the sqrt so it fits in 16 bits.
            std::uint16_t dist;     // This is tuned as a cubic-root so storing it as a cubic-root saves space for free.
        };
        std::uint64_t u64All;
    };

    bool operator ==(const HyperConsts& lhs, const HyperConsts& rhs) noexcept;
    bool operator !=(const HyperConsts& lhs, const HyperConsts& rhs) noexcept;

    // A special case that means "always outside" of the hyperboloid. Essentially, we should never use a metacommand
    // if we see this bit pattern.
    constexpr HyperConsts AlwaysFail = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
    constexpr HyperConsts AlwaysPass = { 0x0000, 0x0000, 0x0000, 0x0000 };

    bool calcHyperboloid(const HyperConsts& consts, const float& c, const float& k, const float& d);
}