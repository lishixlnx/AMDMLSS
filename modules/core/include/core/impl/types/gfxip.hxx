#pragma once

namespace mlss
{

//=====================================================================================================================
//                                          GfxIpTriple
//=====================================================================================================================

struct GfxIpTriple
{
    std::uint32_t stepping : 16;
    std::uint32_t minor : 8;
    std::uint32_t major : 8;

    constexpr GfxIpTriple(const std::uint32_t majorVal = 0, const std::uint32_t minorVal = 0, const std::uint32_t steppingVal = 0)
        : stepping(steppingVal), minor(minorVal), major(majorVal)
    {
    }

    constexpr explicit operator std::uint32_t() const { return stepping | (minor << 16) | (major << 24); }

    explicit operator std::string() const { return std::format("gfx{}{}", major, minor); }
    explicit operator std::wstring() const { return std::format(L"gfx{}{}", major, minor); }
};

struct GfxIpLevel
{
    std::uint32_t reserved : 16;
    std::uint32_t minor    : 8;
    std::uint32_t major    : 8;

    constexpr GfxIpLevel(std::uint32_t major, std::uint32_t minor) : reserved{0}, minor{minor}, major{major} {}

    constexpr GfxIpLevel() : GfxIpLevel(0, 0) {}

    constexpr operator std::uint32_t() const { return reserved | (minor << 16) | (major << 24); }

    explicit operator std::string() const { return std::format("gfx{}{}", major, minor); }
    explicit operator std::wstring() const { return std::format(L"gfx{}{}", major, minor); }
};

struct GfxIpTripleHash
{
    std::size_t operator()(const GfxIpTriple& ip) const noexcept
    {
        return std::hash<std::uint32_t>{}(gfxIpPacked(ip));
    }
};

//=====================================================================================================================
//                                     Comparison & utility
//=====================================================================================================================

[[nodiscard]] constexpr std::uint32_t gfxIpPacked(GfxIpTriple ip) noexcept
{
    return static_cast<std::uint32_t>(ip);
}

[[nodiscard]] constexpr std::uint32_t gfxIpPacked(GfxIpLevel ip) noexcept
{
    return static_cast<std::uint32_t>(ip);
}


constexpr bool operator==(GfxIpTriple lhs, GfxIpTriple rhs) noexcept
{
    return gfxIpPacked(lhs) == gfxIpPacked(rhs);
}

constexpr bool operator!=(GfxIpTriple lhs, GfxIpTriple rhs) noexcept
{
    return gfxIpPacked(lhs) != gfxIpPacked(rhs);
}

constexpr bool operator==(GfxIpTriple lhs, GfxIpLevel rhs) noexcept
{
    return (lhs.major == rhs.major) && (lhs.minor == rhs.minor);
}

constexpr bool operator!=(GfxIpTriple lhs, GfxIpLevel rhs) noexcept
{
    return (lhs.major != rhs.major) || (lhs.minor != rhs.minor);
}

constexpr bool operator==(GfxIpLevel lhs, GfxIpLevel rhs) noexcept
{
    return gfxIpPacked(lhs) == gfxIpPacked(rhs);
}

constexpr bool operator!=(GfxIpLevel lhs, GfxIpLevel rhs) noexcept
{
    return gfxIpPacked(lhs) != gfxIpPacked(rhs);
}

constexpr bool operator==(GfxIpLevel lhs, GfxIpTriple rhs) noexcept
{
    return (lhs.major == rhs.major) && (lhs.minor == rhs.minor);
}

constexpr bool operator!=(GfxIpLevel lhs, GfxIpTriple rhs) noexcept
{
    return (lhs.major != rhs.major) || (lhs.minor != rhs.minor);
}

//=====================================================================================================================
//                                     Well-known GfxIpTriple constants
//=====================================================================================================================

inline constexpr GfxIpTriple IP_GFX_UNKNOWN{0, 0, 0};

// -- Synthetic IPs for generic architecture strings (not hardware-specific) ------------------------------------------

inline constexpr GfxIpTriple IP_GFX9_GENERIC    {255, 1, 0};
inline constexpr GfxIpTriple IP_GFX9_4_GENERIC  {255, 2, 0};
inline constexpr GfxIpTriple IP_GFX10_1_GENERIC {255, 3, 0};
inline constexpr GfxIpTriple IP_GFX10_3_GENERIC {255, 4, 0};
inline constexpr GfxIpTriple IP_GFX11_GENERIC   {255, 5, 0};
inline constexpr GfxIpTriple IP_GFX12_GENERIC   {255, 6, 0};

// -- GFX6 (Southern Islands) -----------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX600{6, 0, 0};
inline constexpr GfxIpTriple IP_GFX601{6, 0, 1};
inline constexpr GfxIpTriple IP_GFX602{6, 0, 2};

// -- GFX7 (Sea Islands) ---------------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX700{7, 0, 0};
inline constexpr GfxIpTriple IP_GFX701{7, 0, 1};
inline constexpr GfxIpTriple IP_GFX702{7, 0, 2};
inline constexpr GfxIpTriple IP_GFX703{7, 0, 3};
inline constexpr GfxIpTriple IP_GFX704{7, 0, 4};
inline constexpr GfxIpTriple IP_GFX705{7, 0, 5};

inline constexpr GfxIpTriple IP_GFX750{7, 5, 0};

// -- GFX8 (Volcanic Islands / Polaris) -------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX801{8, 0, 1};
inline constexpr GfxIpTriple IP_GFX802{8, 0, 2};
inline constexpr GfxIpTriple IP_GFX803{8, 0, 3};
inline constexpr GfxIpTriple IP_GFX805{8, 0, 5};

inline constexpr GfxIpTriple IP_GFX810{8, 1, 0};

// -- GFX9 (Vega / CDNA) ---------------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX900{9, 0, 0};
inline constexpr GfxIpTriple IP_GFX902{9, 0, 2};
inline constexpr GfxIpTriple IP_GFX904{9, 0, 4};
inline constexpr GfxIpTriple IP_GFX905{9, 0, 5};
inline constexpr GfxIpTriple IP_GFX906{9, 0, 6};
inline constexpr GfxIpTriple IP_GFX907{9, 0, 7};
inline constexpr GfxIpTriple IP_GFX908{9, 0, 8};
inline constexpr GfxIpTriple IP_GFX909{9, 0, 9};
inline constexpr GfxIpTriple IP_GFX90a{9, 0, static_cast<std::uint32_t>('a')};
inline constexpr GfxIpTriple IP_GFX90c{9, 0, static_cast<std::uint32_t>('c')};

inline constexpr GfxIpTriple IP_GFX940{9, 4, 0};
inline constexpr GfxIpTriple IP_GFX941{9, 4, 1};
inline constexpr GfxIpTriple IP_GFX942{9, 4, 2};

inline constexpr GfxIpTriple IP_GFX950{9, 5, 0};

// -- GFX10 (RDNA 1 / RDNA 2) ----------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX1000{10, 0, 0};

inline constexpr GfxIpTriple IP_GFX1010{10, 1, 0};
inline constexpr GfxIpTriple IP_GFX1011{10, 1, 1};
inline constexpr GfxIpTriple IP_GFX1012{10, 1, 2};
inline constexpr GfxIpTriple IP_GFX1013{10, 1, 3};

inline constexpr GfxIpTriple IP_GFX1020{10, 2, 0};

inline constexpr GfxIpTriple IP_GFX1030{10, 3, 0};
inline constexpr GfxIpTriple IP_GFX1031{10, 3, 1};
inline constexpr GfxIpTriple IP_GFX1032{10, 3, 2};
inline constexpr GfxIpTriple IP_GFX1033{10, 3, 3};
inline constexpr GfxIpTriple IP_GFX1034{10, 3, 4};
inline constexpr GfxIpTriple IP_GFX1035{10, 3, 5};
inline constexpr GfxIpTriple IP_GFX1036{10, 3, 6};

inline constexpr GfxIpTriple IP_GFX1050{10, 5, 0};

// -- GFX11 (RDNA 3 / RDNA 3.5) --------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX1100{11, 0, 0};
inline constexpr GfxIpTriple IP_GFX1101{11, 0, 1};
inline constexpr GfxIpTriple IP_GFX1102{11, 0, 2};
inline constexpr GfxIpTriple IP_GFX1103{11, 0, 3};
inline constexpr GfxIpTriple IP_GFX1105{11, 0, 5};

inline constexpr GfxIpTriple IP_GFX1150  {11, 5, 0};
inline constexpr GfxIpTriple IP_GFX1151  {11, 5, 1};
inline constexpr GfxIpTriple IP_GFX1152  {11, 5, 2};
inline constexpr GfxIpTriple IP_GFX1153  {11, 5, 3};
inline constexpr GfxIpTriple IP_GFX1154  {11, 5, 4};
inline constexpr GfxIpTriple IP_GFX115FFFF{11, 5, 0xFFFE};
inline constexpr GfxIpTriple IP_GFX115FFFE{11, 5, 0xFFFF};

inline constexpr GfxIpTriple IP_GFX1170{11, 7, 0};
inline constexpr GfxIpTriple IP_GFX1171{11, 7, 1};

// -- GFX12 (RDNA 4) -------------------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX1200  {12, 0, 0};
inline constexpr GfxIpTriple IP_GFX1201  {12, 0, 1};

inline constexpr GfxIpTriple IP_GFX1210  {12, 1, 0};
inline constexpr GfxIpTriple IP_GFX1211  {12, 1, 1};

inline constexpr GfxIpTriple IP_GFX120FFFF{12, 0, 0xFFFE};
inline constexpr GfxIpTriple IP_GFX120FFFE{12, 0, 0xFFFF};

inline constexpr GfxIpTriple IP_GFX1250  {12, 5, 0};
inline constexpr GfxIpTriple IP_GFX1251  {12, 5, 1};

inline constexpr GfxIpTriple IP_GFX125FFFF{12, 5, 0xFFFE};
inline constexpr GfxIpTriple IP_GFX125FFFE{12, 5, 0xFFFF};

// -- GFX13 -----------------------------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX1300   {13, 0, 0};
inline constexpr GfxIpTriple IP_GFX1301   {13, 0, 1};
inline constexpr GfxIpTriple IP_GFX1302   {13, 0, 2};
inline constexpr GfxIpTriple IP_GFX130FFFD{13, 0, 0xFFFD};
inline constexpr GfxIpTriple IP_GFX130FFFF{13, 0, 0xFFFE};
inline constexpr GfxIpTriple IP_GFX130FFFE{13, 0, 0xFFFF};

// -- GFX40 -----------------------------------------------------------------------------------------------------------

inline constexpr GfxIpTriple IP_GFX4000{40, 0, 0};
inline constexpr GfxIpTriple IP_GFX4010{40, 1, 0};
inline constexpr GfxIpTriple IP_GFX4020{40, 2, 0};
inline constexpr GfxIpTriple IP_GFX4030{40, 3, 0};

} // namespace mlss
