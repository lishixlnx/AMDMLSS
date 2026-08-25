/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Analyze embedded Winograd Base gfx1100/fp16 shader archives and test whether
 * each non-reloc ELF exposes a hipModuleLoadData-loadable "main" entry point.
 *
 * Sources (read-only, no AMDMLSS library changes):
 *   modules/shaders/.../Winograd/Base/gfx1100/fp16/shadersBinReloc.hpp
 *   modules/shaders/.../Winograd/Base/gfx1100/fp16/shadersBinNonReloc.hpp
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>

#include "gfx1100/fp16/shadersBinNonReloc.hpp"
#include "gfx1100/fp16/shadersBinReloc.hpp"

using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride1;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dec;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dil;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride1;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dec;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dil;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride1_NonReloc;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dec_NonReloc;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dil_NonReloc;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride1_NonReloc;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dec_NonReloc;
using mlss::conv::mxn::winograd::base::fp16::gfx1100::ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dil_NonReloc;

namespace
{
constexpr std::size_t kElf64EflagsOffset = 0x30U;
constexpr std::uint8_t kEfAmdgpuMachMask = 0xFFU;
constexpr std::uint8_t kMachGfx1100 = 0x041U;

struct GfxMachEntry
{
    const char* gfx_name;
    std::uint8_t mach_code;
};

constexpr GfxMachEntry kGfxMachTable[] = {
    {"gfx1100", 0x041U},
    {"gfx1101", 0x046U},
    {"gfx1102", 0x047U},
    {"gfx1150", 0x043U},
    {"gfx1151", 0x04AU},
    {"gfx1152", 0x055U},
    {"gfx1153", 0x058U},
};

struct Elf64Ehdr
{
    std::uint8_t e_ident[16];
    std::uint16_t e_type;
    std::uint16_t e_machine;
    std::uint32_t e_version;
    std::uint64_t e_entry;
    std::uint64_t e_phoff;
    std::uint64_t e_shoff;
    std::uint32_t e_flags;
    std::uint16_t e_ehsize;
    std::uint16_t e_phentsize;
    std::uint16_t e_phnum;
    std::uint16_t e_shentsize;
    std::uint16_t e_shnum;
    std::uint16_t e_shstrndx;
};

struct Elf64Shdr
{
    std::uint32_t sh_name;
    std::uint32_t sh_type;
    std::uint64_t sh_flags;
    std::uint64_t sh_addr;
    std::uint64_t sh_offset;
    std::uint64_t sh_size;
    std::uint32_t sh_link;
    std::uint32_t sh_info;
    std::uint64_t sh_addralign;
    std::uint64_t sh_entsize;
};

struct Elf64Sym
{
    std::uint32_t st_name;
    std::uint8_t st_info;
    std::uint8_t st_other;
    std::uint16_t st_shndx;
    std::uint64_t st_value;
    std::uint64_t st_size;
};

constexpr std::uint32_t SHT_SYMTAB = 2U;
constexpr std::uint8_t STT_FUNC = 2U;
#define ELF64_ST_TYPE(info) ((info) & 0x0FU)

struct ShaderBlob
{
    const char* archive_name;
    std::span<const std::uint8_t> bytes;
    bool is_reloc;
};

template <std::size_t N>
ShaderBlob make_blob(const char* name, const std::array<std::uint8_t, N>& arr, bool reloc)
{
    return ShaderBlob{name, std::span<const std::uint8_t>(arr.data(), arr.size()), reloc};
}

#define SHADER_BLOB(name, reloc_flag)                                          \
    make_blob(#name, name.m_binary, (reloc_flag))

const ShaderBlob kFp16Gfx1100Shaders[] = {
    SHADER_BLOB(ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride1, true),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dec, true),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dil, true),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride1, true),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dec, true),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dil, true),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride1_NonReloc, false),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dec_NonReloc, false),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F2x3_Fp16Dot2Stride2Dil_NonReloc, false),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride1_NonReloc, false),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dec_NonReloc, false),
    SHADER_BLOB(ConvWinogradElf_Gfx11_F3x2_Fp16Dot2Stride2Dil_NonReloc, false),
};

bool isElfMagic(std::span<const std::uint8_t> image)
{
    return image.size() >= 4U && image[0] == 0x7FU && image[1] == 'E' && image[2] == 'L'
           && image[3] == 'F';
}

bool containsCString(std::span<const std::uint8_t> image, std::string_view needle)
{
    if (needle.empty() || image.size() < needle.size())
        return false;
    for (std::size_t i = 0; i + needle.size() <= image.size(); ++i)
    {
        if (std::memcmp(image.data() + i, needle.data(), needle.size()) == 0)
            return true;
    }
    return false;
}

std::optional<std::uint8_t> machCodeForGfx(std::string_view runtime_gfx)
{
    for (const GfxMachEntry& entry : kGfxMachTable)
    {
        if (runtime_gfx == entry.gfx_name)
            return entry.mach_code;
    }
    return std::nullopt;
}

std::vector<std::uint8_t> patchGfxMetadata(
    std::span<const std::uint8_t> input,
    std::string_view from_gfx,
    std::uint8_t mach_from,
    std::string_view to_gfx,
    std::uint8_t mach_to)
{
    if (from_gfx.size() != to_gfx.size())
        return {};

    std::vector<std::uint8_t> out(input.begin(), input.end());
    if (out.size() > kElf64EflagsOffset && (out[kElf64EflagsOffset] & kEfAmdgpuMachMask) == mach_from)
    {
        out[kElf64EflagsOffset] =
            static_cast<std::uint8_t>((out[kElf64EflagsOffset] & ~kEfAmdgpuMachMask) | mach_to);
    }
    for (std::size_t i = 0; i + from_gfx.size() <= out.size(); ++i)
    {
        if (std::memcmp(out.data() + i, from_gfx.data(), from_gfx.size()) == 0)
            std::memcpy(out.data() + i, to_gfx.data(), to_gfx.size());
    }
    return out;
}

std::vector<std::string> listElfFuncSymbols(std::span<const std::uint8_t> image)
{
    std::vector<std::string> names;
    if (!isElfMagic(image) || image.size() < sizeof(Elf64Ehdr))
        return names;

    const auto* ehdr = reinterpret_cast<const Elf64Ehdr*>(image.data());
    if (ehdr->e_shoff == 0
        || ehdr->e_shoff + static_cast<std::size_t>(ehdr->e_shnum) * sizeof(Elf64Shdr) > image.size())
        return names;

    const auto* shdrs = reinterpret_cast<const Elf64Shdr*>(image.data() + ehdr->e_shoff);
    const Elf64Shdr* symtab = nullptr;
    const Elf64Shdr* strtab = nullptr;

    for (std::uint16_t i = 0; i < ehdr->e_shnum; ++i)
    {
        if (shdrs[i].sh_type == SHT_SYMTAB)
        {
            symtab = &shdrs[i];
            if (symtab->sh_link < ehdr->e_shnum)
                strtab = &shdrs[symtab->sh_link];
            break;
        }
    }
    if (!symtab || !strtab)
        return names;
    if (symtab->sh_offset + symtab->sh_size > image.size()
        || strtab->sh_offset + strtab->sh_size > image.size())
        return names;

    const auto* symbols = reinterpret_cast<const Elf64Sym*>(image.data() + symtab->sh_offset);
    const char* strings = reinterpret_cast<const char*>(image.data() + strtab->sh_offset);
    const std::size_t num_symbols = symtab->sh_size / sizeof(Elf64Sym);

    for (std::size_t i = 0; i < num_symbols; ++i)
    {
        if (symbols[i].st_name >= strtab->sh_size)
            continue;
        const char* sym_name = strings + symbols[i].st_name;
        if (ELF64_ST_TYPE(symbols[i].st_info) != STT_FUNC || symbols[i].st_shndx == 0)
            continue;
        if (sym_name[0] == '\0')
            continue;
        names.emplace_back(sym_name);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

std::string joinSymbols(const std::vector<std::string>& symbols)
{
    std::string out;
    for (std::size_t i = 0; i < symbols.size(); ++i)
    {
        if (i != 0)
            out += ", ";
        out += symbols[i];
    }
    return out.empty() ? "(none)" : out;
}

std::string detectRuntimeGfx()
{
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        return "gfx1100";
    const char* arch = props.gcnArchName;
    const char* colon = std::strchr(arch, ':');
    return std::string(arch, colon ? static_cast<std::size_t>(colon - arch) : std::strlen(arch));
}

struct LoadProbeResult
{
    bool load_ok = false;
    std::string load_error;
    std::string resolved_kernel;
};

LoadProbeResult probeHipLoad(std::span<const std::uint8_t> image, const std::vector<std::string>& candidates)
{
    LoadProbeResult result{};
    hipModule_t module = nullptr;
    const hipError_t load_err = hipModuleLoadData(&module, const_cast<std::uint8_t*>(image.data()));
    if (load_err != hipSuccess)
    {
        result.load_error = hipGetErrorString(load_err);
        return result;
    }
    result.load_ok = true;

    for (const std::string& candidate : candidates)
    {
        hipFunction_t func = nullptr;
        if (hipModuleGetFunction(&func, module, candidate.c_str()) == hipSuccess && func != nullptr)
        {
            result.resolved_kernel = candidate;
            break;
        }
    }

    (void)hipModuleUnload(module);
    return result;
}

void printHeader()
{
    std::cout << "Winograd Base gfx1100/fp16 archived shader analysis\n";
    std::cout << "Runtime GPU: " << detectRuntimeGfx() << '\n';
    std::cout << std::string(120, '-') << '\n';
}

void analyzeShader(const ShaderBlob& shader, std::string_view runtime_gfx)
{
    const auto& image = shader.bytes;
    const std::uint8_t eflags =
        image.size() > kElf64EflagsOffset ? image[kElf64EflagsOffset] : 0U;

    const std::vector<std::string> func_symbols = listElfFuncSymbols(image);
    const bool has_main_cstr = containsCString(image, "main");
    const bool has_amdgpu_cs_main = containsCString(image, "_amdgpu_cs_main");

    std::cout << shader.archive_name << '\n';
    std::cout << "  kind       : " << (shader.is_reloc ? "reloc" : "non-reloc") << '\n';
    std::cout << "  size       : " << image.size() << " bytes\n";
    std::cout << "  elf magic  : " << (isElfMagic(image) ? "7f454c46" : "INVALID") << '\n';
    std::cout << "  eflags@0x30: 0x" << std::hex << static_cast<unsigned>(eflags) << std::dec << '\n';
    std::cout << "  func syms  : " << joinSymbols(func_symbols) << '\n';
    std::cout << "  cstr main  : " << (has_main_cstr ? "yes" : "no") << '\n';
    std::cout << "  cstr _amdgpu_cs_main: " << (has_amdgpu_cs_main ? "yes" : "no") << '\n';

    if (shader.is_reloc)
    {
        std::cout << "  hip load   : skipped (reloc ELF; consumer links offline)\n\n";
        return;
    }

    const auto mach_to = machCodeForGfx(runtime_gfx);
    if (!mach_to.has_value())
    {
        std::cout << "  hip load   : skipped (unknown runtime gfx '" << runtime_gfx << "')\n\n";
        return;
    }

    std::vector<std::uint8_t> patched = patchGfxMetadata(
        image, "gfx1100", kMachGfx1100, runtime_gfx, *mach_to);
    if (patched.empty())
    {
        std::cout << "  hip load   : patch failed\n\n";
        return;
    }

    std::vector<std::string> candidates = {"main", "_amdgpu_cs_main"};
    for (const std::string& sym : func_symbols)
    {
        if (sym.ends_with(".kd"))
            candidates.push_back(sym.substr(0, sym.size() - 3));
        candidates.push_back(sym);
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    const LoadProbeResult probe = probeHipLoad(patched, candidates);
    std::cout << "  patch      : gfx1100 -> " << runtime_gfx << '\n';
    if (!probe.load_ok)
    {
        std::cout << "  hip load   : FAIL (" << probe.load_error << ")\n\n";
        return;
    }

    std::cout << "  hip load   : OK\n";
    if (probe.resolved_kernel.empty())
        std::cout << "  hip func   : NONE of {main, _amdgpu_cs_main, func syms...}\n\n";
    else
        std::cout << "  hip func   : '" << probe.resolved_kernel << "'\n\n";
}

} // namespace

int main()
{
    printHeader();

    const std::string runtime_gfx = detectRuntimeGfx();
    std::size_t non_reloc_count = 0;
    std::size_t load_ok_count = 0;
    std::size_t main_resolved_count = 0;

    for (const ShaderBlob& shader : kFp16Gfx1100Shaders)
    {
        analyzeShader(shader, runtime_gfx);
        if (!shader.is_reloc)
        {
            ++non_reloc_count;
            const auto mach_to = machCodeForGfx(runtime_gfx);
            if (!mach_to.has_value())
                continue;
            const std::vector<std::uint8_t> patched = patchGfxMetadata(
                shader.bytes, "gfx1100", kMachGfx1100, runtime_gfx, *mach_to);
            if (patched.empty())
                continue;

            std::vector<std::string> candidates = {"main", "_amdgpu_cs_main"};
            const std::vector<std::string> func_symbols = listElfFuncSymbols(shader.bytes);
            for (const std::string& sym : func_symbols)
            {
                if (sym.ends_with(".kd"))
                    candidates.push_back(sym.substr(0, sym.size() - 3));
                candidates.push_back(sym);
            }
            const LoadProbeResult probe = probeHipLoad(patched, candidates);
            if (probe.load_ok)
                ++load_ok_count;
            if (probe.resolved_kernel == "main")
                ++main_resolved_count;
        }
    }

    std::cout << std::string(120, '-') << '\n';
    std::cout << "Summary\n";
    std::cout << "  shaders total     : " << std::size(kFp16Gfx1100Shaders) << " (6 reloc + 6 non-reloc)\n";
    std::cout << "  non-reloc tested  : " << non_reloc_count << '\n';
    std::cout << "  hipModuleLoad OK  : " << load_ok_count << '\n';
    std::cout << "  resolved 'main'   : " << main_resolved_count << '\n';
    std::cout << '\n';
    std::cout << "Note: MIGraphX/sample path expects non-reloc + hipModuleGetFunction(\"main\").\n";
    std::cout << "      If func syms show _amdgpu_cs_main but not 'main', the consumer must use\n";
    std::cout << "      the symbol that hipModuleGetFunction actually resolves.\n";

    return (load_ok_count == non_reloc_count && main_resolved_count == non_reloc_count) ? 0 : 1;
}
