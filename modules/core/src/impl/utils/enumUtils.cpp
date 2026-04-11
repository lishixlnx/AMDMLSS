/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "core/core.hpp"

#include <algorithm>
#include <ranges>
#include <format>

namespace mlss
{

    namespace
    {

        struct AsicInfo
        {
            GfxIpTriple gfxIp;
            std::vector<GpuCodenameFlags> codenamesFlags;
            std::string_view archName;
            std::vector<std::string_view> codeNames;
            AsicsTypesFlags asic;
            std::uint8_t elfAscicCode;
            MLSSbool isSupportedHardware;
            std::vector<std::int32_t> numCUs = {};
            std::vector<std::int32_t> mallSizesMB = {};
        };

        template <typename T>
        concept AsicQueryType =
            std::same_as<std::decay_t<T>, AsicInfo> || std::same_as<std::decay_t<T>, GfxIpTriple> ||
            std::same_as<std::decay_t<T>, GpuCodenameFlags> || std::same_as<std::decay_t<T>, std::string_view>;

        template <AsicQueryType LHST, AsicQueryType RHST>
            requires std::is_same_v<LHST, AsicInfo> || std::is_same_v<RHST, AsicInfo>
        constexpr MLSSbool operator==(const LHST& lhs, const RHST& rhs)
        {
            auto compare = []<typename R>(const AsicInfo& l, const R& r) -> bool
            {
                if constexpr (std::is_same_v<AsicInfo, R>)
                {
                    return l.gfxIp == r.gfxIp && l.codenamesFlags == r.codenamesFlags &&
                           l.archName == r.archName && l.codeNames == r.codeNames && l.asic == r.asic &&
                           l.elfAscicCode == r.elfAscicCode && l.isSupportedHardware == r.isSupportedHardware;
                }
                else if constexpr (std::is_same_v<R, GfxIpTriple>)
                {
                    return l.gfxIp == r;
                }
                else if constexpr (std::is_same_v<R, GpuCodenameFlags>)
                {
                    return std::find(l.codenamesFlags.begin(), l.codenamesFlags.end(), r) != l.codenamesFlags.end();
                }
                else if constexpr (std::is_same_v<R, std::string_view>)
                {
                    auto sToLower = [](std::string_view s) -> std::string
                    {
                        std::string result;
                        result.reserve(s.size());
                        for (char c : s)
                        {
                            result.push_back(std::tolower(c));
                        }
                        return result;
                    };

                    std::string tmp_rhs = sToLower(r);
                    std::string tmp_lhs;

                    if (tmp_rhs.find("gfx") != std::string::npos)
                    {
                        tmp_lhs = sToLower(l.archName);

                        return tmp_lhs == tmp_rhs;
                    }
                    else
                    {
                        for (auto codename : l.codeNames)
                        {
                            tmp_lhs = sToLower(codename);

                            if (tmp_lhs == tmp_rhs)
                            {
                                return true;
                            }
                        }

                        return false;
                    }
                }
            }; // compare

            if constexpr (std::is_same_v<LHST, AsicInfo>)
            {
                return compare(lhs, rhs);
            }
            else
            {
                return compare(rhs, lhs);
            }
        }

        template <AsicQueryType LHST, AsicQueryType RHST>
            requires std::is_same_v<LHST, AsicInfo> || std::is_same_v<RHST, AsicInfo>
        constexpr MLSSbool operator!=(const LHST& lhs, const RHST& rhs)
        {
            return !(lhs == rhs);
        }

        static auto asicInfo = std::to_array<AsicInfo>(
            {{.gfxIp = IP_GFX600,
              .codenamesFlags = {GpuCodenameFlags::Tahiti},
              .archName = MLSS_GFX600,
              .codeNames = {MLSS_TAHITI},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x020,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX601,
              .codenamesFlags = {GpuCodenameFlags::Pitcairn, GpuCodenameFlags::Verde},
              .archName = MLSS_GFX601,
              .codeNames = {MLSS_PITCAIRN, MLSS_VERDE},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x021,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX602,
              .codenamesFlags = {GpuCodenameFlags::Oland, GpuCodenameFlags::Hainan},
              .archName = MLSS_GFX602,
              .codeNames = {MLSS_OLAND, MLSS_HAINAN},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x03a,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX700,
              .codenamesFlags = {GpuCodenameFlags::Kaveri, GpuCodenameFlags::Spooky, GpuCodenameFlags::Spectre},
              .archName = MLSS_GFX700,
              .codeNames = {MLSS_KAVERI, MLSS_SPOOKY, MLSS_SPECTRE},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x022,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX701,
              .codenamesFlags = {GpuCodenameFlags::HawaiiPro, GpuCodenameFlags::Grenada},
              .archName = MLSS_GFX701,
              .codeNames = {MLSS_HAWAII_PRO, MLSS_GRENADA},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x023,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX702,
              .codenamesFlags = {GpuCodenameFlags::Hawaii},
              .archName = MLSS_GFX702,
              .codeNames = {MLSS_HAWAII},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x024,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX703,
              .codenamesFlags = {GpuCodenameFlags::Kabini, GpuCodenameFlags::Mullins},
              .archName = MLSS_GFX703,
              .codeNames = {MLSS_KABINI, MLSS_MULLINS},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x025,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX704,
              .codenamesFlags = {GpuCodenameFlags::Bonaire},
              .archName = MLSS_GFX704,
              .codeNames = {MLSS_BONAIRE},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x026,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX705,
              .codenamesFlags = {GpuCodenameFlags::Godavari},
              .archName = MLSS_GFX705,
              .codeNames = {MLSS_GODAVARI},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x03b,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX750,
              .codenamesFlags = {GpuCodenameFlags::Gladius},
              .archName = MLSS_GFX750,
              .codeNames = {MLSS_GLADIUS},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x027,
              .isSupportedHardware = false},

             {.gfxIp = IP_GFX801,
              .codenamesFlags = {GpuCodenameFlags::Carrizo, GpuCodenameFlags::BristolRidge},
              .archName = MLSS_GFX801,
              .codeNames = {MLSS_CARRIZO, MLSS_BRISTOL_RIDGE},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x028,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX802,
              .codenamesFlags = {GpuCodenameFlags::Iceland, GpuCodenameFlags::Tonga},
              .archName = MLSS_GFX802,
              .codeNames = {MLSS_ICELAND, MLSS_TONGA},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x029,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX803,
              .codenamesFlags = {GpuCodenameFlags::Fiji,
                                 GpuCodenameFlags::Polaris10,
                                 GpuCodenameFlags::Polaris11,
                                 GpuCodenameFlags::Polaris12,
                                 GpuCodenameFlags::Polaris22},
              .archName = MLSS_GFX803,
              .codeNames = {MLSS_FIJI, MLSS_POLARIS10, MLSS_POLARIS11, MLSS_POLARIS12, MLSS_POLARIS22},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x02a,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX805,
              .codenamesFlags = {GpuCodenameFlags::TongaPro},
              .archName = MLSS_GFX805,
              .codeNames = {MLSS_TONGA_PRO},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x03c,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX810,
              .codenamesFlags = {GpuCodenameFlags::StoneyRidge},
              .archName = MLSS_GFX810,
              .codeNames = {MLSS_STONEY_RIDGE},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x02b,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX900,
              .codenamesFlags = {GpuCodenameFlags::Vega10, GpuCodenameFlags::Greenland},
              .archName = MLSS_GFX900,
              .codeNames = {MLSS_VEGA10, MLSS_GREENLAND},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x02c,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX902,
              .codenamesFlags = {GpuCodenameFlags::RavenRidge},
              .archName = MLSS_GFX902,
              .codeNames = {MLSS_RAVEN_RIDGE},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x02d,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX904,
              .codenamesFlags = {GpuCodenameFlags::Vega12},
              .archName = MLSS_GFX904,
              .codeNames = {MLSS_VEGA12},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x02e,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX906,
              .codenamesFlags = {GpuCodenameFlags::Vega20},
              .archName = MLSS_GFX906,
              .codeNames = {MLSS_VEGA20},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x02f,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX908,
              .codenamesFlags = {GpuCodenameFlags::Arcturus, GpuCodenameFlags::MI100},
              .archName = MLSS_GFX908,
              .codeNames = {MLSS_ARCTURUS, MLSS_MI100},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x030,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX909,
              .codenamesFlags = {GpuCodenameFlags::RavenRidge2},
              .archName = MLSS_GFX909,
              .codeNames = {MLSS_RAVEN_RIDGE2},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x031,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX90a,
              .codenamesFlags = {GpuCodenameFlags::Aldebaran, GpuCodenameFlags::MI200},
              .archName = MLSS_GFX90A,
              .codeNames = {MLSS_ALDEBARAN, MLSS_MI200},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x03F,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX90c,
              .codenamesFlags = {GpuCodenameFlags::Renoir},
              .archName = MLSS_GFX90C,
              .codeNames = {MLSS_RENOIR},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x032,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX940,
              .codenamesFlags = {GpuCodenameFlags::MI300A},
              .archName = MLSS_GFX940,
              .codeNames = {MLSS_MI300A},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x040,
              .isSupportedHardware = false,
              .numCUs = {228}},
             {.gfxIp = IP_GFX941,
              .codenamesFlags = {GpuCodenameFlags::MI300X},
              .archName = MLSS_GFX941,
              .codeNames = {MLSS_MI300X},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x04B,
              .isSupportedHardware = false,
              .numCUs = {304}},
             {.gfxIp = IP_GFX942,
              .codenamesFlags = {GpuCodenameFlags::AquaVanjaram, GpuCodenameFlags::MI325X},
              .archName = MLSS_GFX942,
              .codeNames = {MLSS_AQUA_VANJARAM, MLSS_MI325X},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x04C,
              .isSupportedHardware = false,
              .numCUs = {64, 80, 228, 304}}, // 228 is for some gfx942A1, other A1 variants had 304 CUs
             {.gfxIp = IP_GFX950,
              .codenamesFlags = {GpuCodenameFlags::MI350},
              .archName = MLSS_GFX950,
              .codeNames = {MLSS_MI350},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x04F,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1000,
              .codenamesFlags = {GpuCodenameFlags::Navi10Lite, GpuCodenameFlags::Ariel},
              .archName = MLSS_GFX1000,
              .codeNames = {MLSS_NAVI10, MLSS_ARIEL},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x0F1,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1010,
              .codenamesFlags = {GpuCodenameFlags::Navi10},
              .archName = MLSS_GFX1010,
              .codeNames = {MLSS_NAVI10},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x033,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1011,
              .codenamesFlags = {GpuCodenameFlags::Navi12},
              .archName = MLSS_GFX1011,
              .codeNames = {MLSS_NAVI10}, // Using NAVI10 as fallback since MLSS_NAVI12 is not defined
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x034,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1012,
              .codenamesFlags = {GpuCodenameFlags::Navi14},
              .archName = MLSS_GFX1012,
              .codeNames = {MLSS_NAVI14},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x035,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1013,
              .codenamesFlags = {GpuCodenameFlags::Robin},
              .archName = MLSS_GFX1013,
              .codeNames = {MLSS_ROBIN},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x042,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1020,
              .codenamesFlags = {GpuCodenameFlags::Arden, GpuCodenameFlags::Navi21Lite},
              .archName = MLSS_GFX1030,   // Using GFX1030 as fallback since MLSS_GFX1020 is not defined
              .codeNames = {MLSS_NAVI21}, // Using available macro
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x0FD,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1030,
              .codenamesFlags = {GpuCodenameFlags::Navi21, GpuCodenameFlags::SiennaCichlid},
              .archName = MLSS_GFX1030,
              .codeNames = {MLSS_NAVI21, MLSS_SIENNA_CICHLID},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x036,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1031,
              .codenamesFlags = {GpuCodenameFlags::Navi22, GpuCodenameFlags::NavyFlounder},
              .archName = MLSS_GFX1031,
              .codeNames = {MLSS_NAVI22}, // Removed undefined MLSS_NAVY_FLOUNDER
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x037,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1032,
              .codenamesFlags = {GpuCodenameFlags::Navi23, GpuCodenameFlags::DimgrayCaveFish},
              .archName = MLSS_GFX1032,
              .codeNames = {MLSS_NAVI23, MLSS_DIMGREY_CAVEFISH},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x038,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1033,
              .codenamesFlags = {GpuCodenameFlags::VanGogh, GpuCodenameFlags::Mero},
              .archName = MLSS_GFX1034, // Using GFX1034 as fallback since MLSS_GFX1033 is not defined
              .codeNames = {MLSS_VAN_GOGH, MLSS_NERO},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x039,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1034,
              .codenamesFlags = {GpuCodenameFlags::Navi24},
              .archName = MLSS_GFX1034,
              .codeNames = {MLSS_NAVI24},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x03E,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1035,
              .codenamesFlags = {GpuCodenameFlags::Rembrandt},
              .archName = MLSS_GFX1035,
              .codeNames = {MLSS_REMBRANDT},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x03D,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1036,
              .codenamesFlags = {GpuCodenameFlags::Raphael, GpuCodenameFlags::Mendocino},
              .archName = MLSS_GFX1036,
              .codeNames = {MLSS_RAPHAEL, MLSS_MENDOCINO},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x045,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1050,
              .codenamesFlags = {GpuCodenameFlags::Viola},
              .archName = MLSS_GFX1050,
              .codeNames = {MLSS_VIOLA},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0FC,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1100,
              .codenamesFlags = {GpuCodenameFlags::Navi31},
              .archName = MLSS_GFX1100,
              .codeNames = {MLSS_NAVI31},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x041,
              .isSupportedHardware = true,
              .numCUs = {24, 70, 72, 96, 84},
              .mallSizesMB = {80, 96}}, // Navi31 XTX and XT-W had 24/96 CUs, XLW had 70CUs
             {.gfxIp = IP_GFX1101,
              .codenamesFlags = {GpuCodenameFlags::Navi32},
              .archName = MLSS_GFX1101,
              .codeNames = {MLSS_NAVI32},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x046,
              .isSupportedHardware = true,
              .numCUs = {48, 54, 60},
              .mallSizesMB = {64}},
             {.gfxIp = IP_GFX1102,
              .codenamesFlags = {GpuCodenameFlags::Navi33},
              .archName = MLSS_GFX1102,
              .codeNames = {MLSS_NAVI33},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x047,
              .isSupportedHardware = true,
              .numCUs = {16, 32},
              .mallSizesMB = {32}},
             {.gfxIp = IP_GFX1103,
              .codenamesFlags = {GpuCodenameFlags::Phoenix, GpuCodenameFlags::Phoenix2},
              .archName = MLSS_GFX1103,
              .codeNames = {MLSS_PHOENIX, MLSS_PHOENIX2},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x044,
              .isSupportedHardware = true,
              .numCUs = {4, 8, 12}},
             {.gfxIp = IP_GFX1105,
              .codenamesFlags = {GpuCodenameFlags::Navi32GLXL},
              .archName = MLSS_GFX1105,
              .codeNames = {MLSS_NAVI32GLXL},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x04D,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1150,
              .codenamesFlags = {GpuCodenameFlags::Strix1,
                                 GpuCodenameFlags::Strix2,
                                 GpuCodenameFlags::Strix3,
                                 GpuCodenameFlags::GorgonPoint1},
              .archName = MLSS_GFX1150,
              .codeNames = {MLSS_STRIX1, MLSS_STRIX2, MLSS_STRIX3, MLSS_GORGON_POINT1},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x043,
              .isSupportedHardware = true,
              .numCUs = {16}},
             {.gfxIp = IP_GFX1151,
              .codenamesFlags = {GpuCodenameFlags::Sarlak, GpuCodenameFlags::StrixHalo},
              .archName = MLSS_GFX1151,
              .codeNames = {MLSS_SARLAK, MLSS_STRIX_HALO},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x04A,
              .isSupportedHardware = true,
              .numCUs = {16, 32, 40},
              .mallSizesMB = {32}},
             {.gfxIp = IP_GFX1152,
              .codenamesFlags = {GpuCodenameFlags::Krackan, GpuCodenameFlags::GorgonPoint2},
              .archName = MLSS_GFX1152,
              .codeNames = {MLSS_KRACKAN, MLSS_GORGON_POINT2},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x055,
              .isSupportedHardware = true,
              .numCUs = {8}},
             {.gfxIp = IP_GFX1153,
              .codenamesFlags = {GpuCodenameFlags::Krackan2e, GpuCodenameFlags::GorgonPoint3},
              .archName = MLSS_GFX1153,
              .codeNames = {MLSS_KRACKAN2E, MLSS_GPT3},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x058,
              .isSupportedHardware = true,
              .numCUs = {4}},
             {.gfxIp = IP_GFX1154,
              .codenamesFlags = {GpuCodenameFlags::Medusa3},
              .archName = MLSS_GFX1154,
              .codeNames = {MLSS_MEDUSA3},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x05C,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX115FFFF,
              .codenamesFlags = {GpuCodenameFlags::Strix1A0},
              .archName = MLSS_GFX115FFFF,
              .codeNames = {MLSS_STRIX1_A0},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x0F7,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX115FFFE,
              .codenamesFlags = {GpuCodenameFlags::Medusa1A0},
              .archName = MLSS_GFX115FFFE,
              .codeNames = {MLSS_MEDUSA1_A0},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x0F0,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1170,
              .codenamesFlags = {GpuCodenameFlags::Medusa1B0},
              .archName = MLSS_GFX1170,
              .codeNames = {MLSS_MEDUSA1_B0},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x05D,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1171,
              .codenamesFlags = {GpuCodenameFlags::Medusa2},
              .archName = MLSS_GFX1171,
              .codeNames = {MLSS_MEDUSA2},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x05E,
              .isSupportedHardware = true,
              .numCUs = {48, 56, 64}},
             {.gfxIp = IP_GFX1200,
              .codenamesFlags = {GpuCodenameFlags::Navi44},
              .archName = MLSS_GFX1200,
              .codeNames = {MLSS_NAVI44},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x048,
              .isSupportedHardware = true,
              .numCUs = {28, 32},
              .mallSizesMB = {32}},
             {.gfxIp = IP_GFX1201,
              .codenamesFlags = {GpuCodenameFlags::Navi48},
              .archName = MLSS_GFX1201,
              .codeNames = {MLSS_NAVI48},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x04E,
              .isSupportedHardware = true,
              .numCUs = {64},
              .mallSizesMB = {64}},
             {.gfxIp = IP_GFX1210,
              .codenamesFlags = {GpuCodenameFlags::MI400XCDML},
              .archName = MLSS_GFX1210,
              .codeNames = {MLSS_MI400XCDML},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x049,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1211,
              .codenamesFlags = {GpuCodenameFlags::MI400XCDGP},
              .archName = MLSS_GFX1211,
              .codeNames = {MLSS_MI400XCDGP},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x5A,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX120FFFF,
              .codenamesFlags = {GpuCodenameFlags::Navi44A0},
              .archName = MLSS_GFX120FFFF,
              .codeNames = {MLSS_NAVI44_A0},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x0F4,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX120FFFE,
              .codenamesFlags = {GpuCodenameFlags::Navi48A0},
              .archName = MLSS_GFX120FFFE,
              .codeNames = {MLSS_NAVI48_A0},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x0F3,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1250,
              .codenamesFlags = {},
              .archName = MLSS_GFX1250,
              .codeNames = {},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x049,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1251,
              .codenamesFlags = {},
              .archName = MLSS_GFX1251,
              .codeNames = {},
              .asic = AsicsTypesFlags::APU,
              .elfAscicCode = 0x05a,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX1300,
              .codenamesFlags = {GpuCodenameFlags::AlphaTrion2},
              .archName = MLSS_GFX1300,
              .codeNames = {MLSS_ALPHA_TRION_2},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x050,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1301,
              .codenamesFlags = {GpuCodenameFlags::AlphaTrion1},
              .archName = MLSS_GFX1301,
              .codeNames = {MLSS_ALPHA_TRION_1},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x056,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX1302,
              .codenamesFlags = {GpuCodenameFlags::AlphaTrion3},
              .archName = MLSS_GFX1302,
              .codeNames = {MLSS_ALPHA_TRION_3},
              .asic = AsicsTypesFlags::dGPU,
              .elfAscicCode = 0x057,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX130FFFF,
              .codenamesFlags = {GpuCodenameFlags::Canis},
              .archName = MLSS_GFX130FFFF,
              .codeNames = {MLSS_CANIS},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0F5,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX130FFFE,
              .codenamesFlags = {GpuCodenameFlags::Magnus},
              .archName = MLSS_GFX130FFFE,
              .codeNames = {MLSS_MAGNUS},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0F2,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX130FFFD,
              .codenamesFlags = {GpuCodenameFlags::Orion},
              .archName = MLSS_GFX130FFFD,
              .codeNames = {MLSS_ORION},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0F1,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX4000,
              .codenamesFlags = {GpuCodenameFlags::Mariner, GpuCodenameFlags::Mobile0, GpuCodenameFlags::VanGoghLite},
              .archName = MLSS_GFX4000,
              .codeNames = {MLSS_MARINER, MLSS_MOBILE0, MLSS_VAN_GOGH_LITE},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0F8,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX4010,
              .codenamesFlags = {GpuCodenameFlags::Viking, GpuCodenameFlags::Mobile1, GpuCodenameFlags::MGFX1},
              .archName = MLSS_GFX4010,
              .codeNames = {MLSS_VIKING, MLSS_MOBILE1, MLSS_MGFX1},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0F9,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX4020,
              .codenamesFlags = {GpuCodenameFlags::Mobile2EVT1plus},
              .archName = MLSS_GFX4020,
              .codeNames = {MLSS_MOBILE2_EVT1PLUS},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0FE,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX4030,
              .codenamesFlags = {GpuCodenameFlags::Mobile3},
              .archName = MLSS_GFX4030,
              .codeNames = {MLSS_MOBILE3},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x0F6,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX9_GENERIC,
              .codenamesFlags = {},
              .archName = MLSS_GFX9_GENERIC,
              .codeNames = {},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x051,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX9_4_GENERIC,
              .codenamesFlags = {},
              .archName = MLSS_GFX9_4_GENERIC,
              .codeNames = {},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x05F,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX10_1_GENERIC,
              .codenamesFlags = {},
              .archName = MLSS_GFX10_1_GENERIC,
              .codeNames = {},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x052,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX10_3_GENERIC,
              .codenamesFlags = {},
              .archName = MLSS_GFX10_3_GENERIC,
              .codeNames = {},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x053,
              .isSupportedHardware = false},
             {.gfxIp = IP_GFX11_GENERIC,
              .codenamesFlags = {},
              .archName = MLSS_GFX11_GENERIC,
              .codeNames = {},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x054,
              .isSupportedHardware = true},
             {.gfxIp = IP_GFX12_GENERIC,
              .codenamesFlags = {},
              .archName = MLSS_GFX12_GENERIC,
              .codeNames = {},
              .asic = AsicsTypesFlags::unknown,
              .elfAscicCode = 0x059,
              .isSupportedHardware = true}});
    } // namespace

    //=====================================================================================================================
    enum64 makeArrayEnum(const std::uint32_t& type, const size_t& size)
    {
        return {(static_cast<uint64_t>((size > 1) && !(type & MLSS_ARRAY) ? type | MLSS_ARRAY : type) << 32) | size};
    }

    //=====================================================================================================================
    size_t getArrayEnumSize(const enum64& type)
    {
        return static_cast<size_t>(type.m_attr.u64 & 0xFFFFFFFF);
    }

    //=====================================================================================================================
    std::uint32_t getArrayEnumType(const enum64& type)
    {
        return static_cast<std::uint32_t>((type.m_attr.u64 >> 32) & 0xFFFFFFFF);
    }

    //=====================================================================================================================
    std::uint32_t getArrayEnumElementType(const enum64& type)
    {
        return getArrayEnumType(type) & MLSS_TYPE_MASK;
    }

    //=====================================================================================================================
    enum64 getFlagFromString(const std::string& src)
    {
        if (src == "bool")
        {
            return makeArrayEnum(MLSS_BOOL);
        }
        else if (src == "int8")
        {
            return makeArrayEnum(MLSS_INT8);
        }
        else if (src == "uint8")
        {
            return makeArrayEnum(MLSS_UINT8);
        }
        else if (src == "int16")
        {
            return makeArrayEnum(MLSS_INT16);
        }
        else if (src == "uint16")
        {
            return makeArrayEnum(MLSS_UINT16);
        }
        else if (src == "int32")
        {
            return makeArrayEnum(MLSS_INT32);
        }
        else if (src == "uint32")
        {
            return makeArrayEnum(MLSS_UINT32);
        }
        else if (src == "int64")
        {
            return makeArrayEnum(MLSS_INT64);
        }
        else if (src == "uint64")
        {
            return makeArrayEnum(MLSS_UINT64);
        }
        else if (src == "float32")
        {
            return makeArrayEnum(MLSS_FLOAT32);
        }
        else if (src == "float64")
        {
            return makeArrayEnum(MLSS_FLOAT64);
        }
        else if (src == "float4")
        {
            return makeArrayEnum(MLSS_FLOAT4);
        }
        else if (src == "float8")
        {
            return makeArrayEnum(MLSS_FLOAT8);
        }
        else if (src == "float8_fnuz")
        {
            return makeArrayEnum(MLSS_FLOAT8_FNUZ);
        }
        else if (src == "float8_ocp")
        {
            return makeArrayEnum(MLSS_FLOAT8_OCP);
        }
        else if (src == "float16")
        {
            return makeArrayEnum(MLSS_FLOAT16);
        }
        else if (src == "bfloat4")
        {
            return makeArrayEnum(MLSS_BFLOAT4);
        }
        else if (src == "bfloat8")
        {
            return makeArrayEnum(MLSS_BFLOAT8);
        }
        else if (src == "bfloat8_fnuz")
        {
            return makeArrayEnum(MLSS_BFLOAT8_FNUZ);
        }
        else if (src == "bfloat8_ocp")
        {
            return makeArrayEnum(MLSS_BFLOAT8_OCP);
        }
        else if (src == "bfloat16")
        {
            return makeArrayEnum(MLSS_BFLOAT16);
        }
        else if (src == "enum")
        {
            return makeArrayEnum(MLSS_ENUM);
        }
        else if (src == "array")
        {
            return makeArrayEnum(MLSS_ARRAY);
        }
        else
        {
            return makeArrayEnum(MLSS_UNKNOWN_TYPE);
        }
    }

    //=====================================================================================================================
    std::expected<GfxIpTriple, std::error_code> gpuCodenameToGfxIpTriple(GpuCodenameFlags codename) noexcept
    {
        if (auto it = std::find(asicInfo.begin(), asicInfo.end(), codename); it != asicInfo.end())
        {
            if (it->isSupportedHardware)
            {
                return it->gfxIp;
            }
            else
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
            }
        }
        return std::unexpected(make_error_code(MLSSErrorCode::CodenameNotFound));
    }

    //=====================================================================================================================
    std::expected<std::string_view, std::error_code> gfxIpTripleToString(GfxIpTriple gfxIp) noexcept
    {
        if (auto it = std::find(asicInfo.begin(), asicInfo.end(), gfxIp); it != asicInfo.end())
        {
            if (it->isSupportedHardware)
            {
                return it->archName;
            }
            else
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
            }
        }
        return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
    }

    //=====================================================================================================================
    std::expected<std::string_view, std::error_code> gpuCodenameFlagsToString(GpuCodenameFlags flag) noexcept
    {
        for (const auto& info : asicInfo)
        {
            auto it = std::find(info.codenamesFlags.begin(), info.codenamesFlags.end(), flag);
            if (it != info.codenamesFlags.end())
            {
                if (info.isSupportedHardware)
                {
                    auto idx = std::distance(info.codenamesFlags.begin(), it);
                    if (idx < info.codeNames.size())
                    {
                        return info.codeNames[idx];
                    }
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
                }
            }
        }
        return std::unexpected(make_error_code(MLSSErrorCode::CodenameNotFound));
    }

    //=====================================================================================================================
    std::expected<GfxIpTriple, std::error_code> architectureStringToGfxIpTriple(std::string_view gfx) noexcept
    {
        auto trim = [](std::string_view value) -> std::string_view
        {
            const auto first = value.find_first_not_of(" \t\r\n");
            if (first == std::string_view::npos)
            {
                return {};
            }
            const auto last = value.find_last_not_of(" \t\r\n");
            return value.substr(first, last - first + 1);
        };

        std::string normalized;
        if (auto trimmed = trim(gfx); !trimmed.empty())
        {
            normalized.reserve(trimmed.size() + 10);
            std::ranges::transform(trimmed, std::back_inserter(normalized), [](unsigned char ch)
                                   { return static_cast<char>(std::toupper(ch)); });

            if (!normalized.starts_with("MLSS_"))
            {
                if (normalized.starts_with("GFX"))
                {
                    normalized.insert(0, "MLSS_");
                }
                else
                {
                    normalized.insert(0, "MLSS_GFX");
                }
            }
        }

        if (normalized.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
        }

        auto it = std::ranges::find_if(asicInfo, [&](const AsicInfo& info)
                                       { return info.archName == normalized; });

        if (it == asicInfo.end())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
        }

        if (!it->isSupportedHardware)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
        }

        return it->gfxIp;
    }

    //=====================================================================================================================
    std::expected<GpuCodenameFlags, std::error_code> gpuCodenameStringToFlag(std::string_view codename) noexcept
    {
        for (const auto& info : asicInfo)
        {
            for (size_t i = 0; i < info.codeNames.size(); ++i)
            {
                if (info.codeNames[i] == codename && i < info.codenamesFlags.size())
                {
                    if (info.isSupportedHardware)
                    {
                        return info.codenamesFlags[i];
                    }
                    else
                    {
                        return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
                    }
                }
            }
        }
        return std::unexpected(make_error_code(MLSSErrorCode::CodenameNotFound));
    }

    //=====================================================================================================================
    std::expected<GfxIpTriple, std::error_code> elfMatchToGfxIpTriple(const std::uint8_t& elfMatch) noexcept
    {
        for (const auto& info : asicInfo)
        {
            if (info.elfAscicCode == elfMatch)
            {
                if (info.isSupportedHardware)
                {
                    return info.gfxIp;
                }
                else
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
                }
            }
        }
        return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
    }

    //=====================================================================================================================
    std::expected<std::uint8_t, std::error_code> gfxIpTripleToElfMatch(GfxIpTriple gfxIp) noexcept
    {
        if (auto it = std::find(asicInfo.begin(), asicInfo.end(), gfxIp); it != asicInfo.end())
        {
            if (it->isSupportedHardware)
            {
                return it->elfAscicCode;
            }
            else
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
            }
        }
        return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
    }

    //=====================================================================================================================
    bool isShaderSupported(ShaderTypesFlags shaderType, GfxIpTriple gfxIp) noexcept
    {
        const auto p = gfxIpPacked(gfxIp);
        switch (shaderType)
        {
            case ShaderTypesFlags::WMMA:
                // WMMA supported on RDNA3 (Gfx110x), RDNA3.5 (Gfx115x), and RDNA4 (Gfx120x)
                return ((p >= gfxIpPacked(IP_GFX1100)) && (p <= gfxIpPacked(IP_GFX1103))) ||
                       ((p >= gfxIpPacked(IP_GFX1150)) && (p <= gfxIpPacked(IP_GFX1154))) ||
                       ((p >= gfxIpPacked(IP_GFX1200)) && (p <= gfxIpPacked(IP_GFX1201)));

            case ShaderTypesFlags::XDL:
                // XDL supported on CDNA/MI accelerators (Gfx908, Gfx90a, Gfx940-942, Gfx950)
                return (gfxIp == IP_GFX908) ||
                       (gfxIp == IP_GFX90a) ||
                       ((p >= gfxIpPacked(IP_GFX940)) && (p <= gfxIpPacked(IP_GFX942))) ||
                       (gfxIp == IP_GFX950);

            case ShaderTypesFlags::DL:
                // DL (dot product) supported on Vega20+ (Gfx906 and newer)
                return (p >= gfxIpPacked(IP_GFX906));

            default:
                return false;
        }
    }

    //=====================================================================================================================
    std::expected<GfxIpTriple, std::error_code> getHighEndGfxIpTriple(GfxIpTriple gfxIp) noexcept
    {
        const auto p = gfxIpPacked(gfxIp);

        // GCN 1.0 / Southern Islands (Gfx6xx) -> Gfx600 (Tahiti)
        if ((p >= gfxIpPacked(IP_GFX600)) && (p <= gfxIpPacked(IP_GFX602)))
        {
            return IP_GFX600;
        }

        // GCN 2.0 / Sea Islands (Gfx7xx) -> Gfx702 (Hawaii)
        if ((p >= gfxIpPacked(IP_GFX700)) && (p <= gfxIpPacked(IP_GFX750)))
        {
            return IP_GFX702;
        }

        // GCN 3.0/4.0 / Volcanic Islands & Polaris (Gfx8xx) -> Gfx803 (Fiji/Polaris)
        if ((p >= gfxIpPacked(IP_GFX801)) && (p <= gfxIpPacked(IP_GFX810)))
        {
            return IP_GFX803;
        }

        // GCN 5.0 / Vega (Gfx900-906) -> Gfx906 (Vega20)
        if ((p >= gfxIpPacked(IP_GFX900)) && (p <= gfxIpPacked(IP_GFX906)))
        {
            return IP_GFX906;
        }

        // CDNA 1 (Gfx908) -> Gfx908 (MI100)
        if (gfxIp == IP_GFX908)
        {
            return IP_GFX908;
        }

        // Vega APUs (Gfx909, Gfx90c) -> Gfx90c (Renoir)
        if ((gfxIp == IP_GFX909) || (gfxIp == IP_GFX90c))
        {
            return IP_GFX90c;
        }

        // CDNA 2 family (Gfx90a) -> Gfx90a (MI200)
        if (gfxIp == IP_GFX90a)
        {
            return IP_GFX90a;
        }

        // CDNA 3 family (Gfx94x) -> Gfx942 (MI300)
        if ((p >= gfxIpPacked(IP_GFX940)) && (p <= gfxIpPacked(IP_GFX942)))
        {
            return IP_GFX942;
        }

        // CDNA 3.5 (Gfx950) -> Gfx950 (MI350)
        if (gfxIp == IP_GFX950)
        {
            return IP_GFX950;
        }

        // RDNA 1 (Gfx10xx) -> Gfx1010 (Navi10)
        if ((p >= gfxIpPacked(IP_GFX1000)) && (p <= gfxIpPacked(IP_GFX1013)))
        {
            return IP_GFX1010;
        }

        // RDNA 2 (Gfx103x) -> Gfx1030 (Navi21)
        if ((p >= gfxIpPacked(IP_GFX1020)) && (p <= gfxIpPacked(IP_GFX1050)))
        {
            return IP_GFX1030;
        }

        // RDNA 3 family (Gfx11xx) -> Gfx1100 (Navi31)
        if ((p >= gfxIpPacked(IP_GFX1100)) && (p <= gfxIpPacked(IP_GFX1105)))
        {
            return IP_GFX1100;
        }

        // RDNA 3.5 family (Gfx115x) -> Gfx1150 (Strix)
        if ((p >= gfxIpPacked(IP_GFX1150)) && (p <= gfxIpPacked(IP_GFX1154)))
        {
            return IP_GFX1150;
        }

        // RDNA 3.5 A0 variants -> Gfx1150
        if ((gfxIp == IP_GFX115FFFF) || (gfxIp == IP_GFX115FFFE))
        {
            return IP_GFX1150;
        }

        // RDNA 3.5 Medusa family (Gfx117x) -> Gfx1170
        if ((p >= gfxIpPacked(IP_GFX1170)) && (p <= gfxIpPacked(IP_GFX1171)))
        {
            return IP_GFX1170;
        }

        // RDNA 4 family (Gfx120x) -> Gfx1201 (Navi48 - higher-end than Navi44)
        if ((p >= gfxIpPacked(IP_GFX1200)) && (p <= gfxIpPacked(IP_GFX1211)))
        {
            return IP_GFX1201;
        }

        // RDNA 4 A0 variants -> Gfx1201
        if ((gfxIp == IP_GFX120FFFF) || (gfxIp == IP_GFX120FFFE))
        {
            return IP_GFX1201;
        }

        // RDNA 4 Gfx125x family -> Gfx1250
        if ((p >= gfxIpPacked(IP_GFX1250)) && (p <= gfxIpPacked(IP_GFX1251)))
        {
            return IP_GFX1250;
        }

        // RDNA 5 / Gfx13xx family -> Gfx1300
        if ((p >= gfxIpPacked(IP_GFX1300)) && (p <= gfxIpPacked(IP_GFX1302)))
        {
            return IP_GFX1300;
        }

        // Gfx13 pre-release variants -> Gfx1300
        if ((gfxIp == IP_GFX130FFFF) || (gfxIp == IP_GFX130FFFE) || (gfxIp == IP_GFX130FFFD))
        {
            return IP_GFX1300;
        }

        // Gfx4xxx mobile/custom (Gfx40xx) -> Gfx4000
        if ((p >= gfxIpPacked(IP_GFX4000)) && (p <= gfxIpPacked(IP_GFX4030)))
        {
            return IP_GFX4000;
        }

        // Generic architectures return themselves
        if (gfxIp == IP_GFX9_GENERIC || gfxIp == IP_GFX9_4_GENERIC || gfxIp == IP_GFX10_1_GENERIC ||
            gfxIp == IP_GFX10_3_GENERIC || gfxIp == IP_GFX11_GENERIC || gfxIp == IP_GFX12_GENERIC)
        {
            return gfxIp;
        }

        // Unknown architecture
        if (gfxIp == IP_GFX_UNKNOWN)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
        }

        // For any unhandled architectures, return an error
        return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
    }

    //=====================================================================================================================
    std::expected<std::int32_t, std::error_code> getNumCu(GfxIpTriple gfxIp) noexcept
    {
        auto it = std::find(asicInfo.begin(), asicInfo.end(), gfxIp);

        if (it == asicInfo.end())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
        }

        if (!it->isSupportedHardware)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
        }

        if (it->numCUs.empty())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
        }

        return *std::max_element(it->numCUs.begin(), it->numCUs.end());
    }

    //=====================================================================================================================
    std::expected<std::int32_t, std::error_code> getMALL(GfxIpTriple gfxIp) noexcept
    {
        auto it = std::find(asicInfo.begin(), asicInfo.end(), gfxIp);

        if (it == asicInfo.end())
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotFound));
        }

        if (!it->isSupportedHardware)
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
        }

        if (it->mallSizesMB.empty())
        {
            return 0;
        }

        return *std::max_element(it->mallSizesMB.begin(), it->mallSizesMB.end());
    }

} // namespace mlss
