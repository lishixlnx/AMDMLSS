/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "core/core.hpp"

#include <amd_comgr/amd_comgr.h>
#include <cstring>

namespace mlss
{

    namespace
    {
        // ELF64 structures for parsing AMD GPU code objects
        struct Elf64_Ehdr
        {
            unsigned char e_ident[16];
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

        struct Elf64_Shdr
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

        struct Elf64_Sym
        {
            std::uint32_t st_name;
            std::uint8_t st_info;
            std::uint8_t st_other;
            std::uint16_t st_shndx;
            std::uint64_t st_value;
            std::uint64_t st_size;
        };

        // ELF constants
        constexpr std::uint32_t SHT_SYMTAB = 0x02;
        constexpr std::uint32_t SHT_STRTAB = 0x03;
        constexpr std::uint8_t STT_FUNC = 0x02;
        constexpr std::uint8_t STT_OBJECT = 0x01;
        constexpr std::size_t ELF64_E_FLAGS_OFFSET = 0x30;
        constexpr std::uint8_t EF_AMDGPU_MACH_MASK = 0xFF;

        // Extract symbol type from st_info
        constexpr std::uint8_t ELF64_ST_TYPE(std::uint8_t info)
        {
            return info & 0xf;
        }

        // Check if string ends with suffix
        bool endsWith(const std::string& str, const std::string& suffix)
        {
            if (suffix.size() > str.size()) return false;
            return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        }



        std::expected<std::string, std::error_code> gfxIpToIsaString(GfxIpTriple gfxIp)
        {
            auto archNameResult = gfxIpTripleToString(gfxIp);
            if (!archNameResult.has_value())
            {
                return std::unexpected(archNameResult.error());
            }

            std::string_view archName = archNameResult.value();
            constexpr std::string_view mlssPrefix = "MLSS_";
            if (archName.starts_with(mlssPrefix))
            {
                archName.remove_prefix(mlssPrefix.size());
            }

            std::string result;
            result.reserve(archName.size());
            for (char c : archName)
            {
                result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }

            return result;
        }

        std::expected<std::vector<std::uint8_t>, std::error_code> patchGfxVersion(
            std::span<const std::uint8_t> input,
            std::string_view from, std::uint8_t machFrom,
            std::string_view to,   std::uint8_t machTo)
        {
            if (from.size() != to.size())
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
            }

            std::vector<std::uint8_t> patched(input.begin(), input.end());

            if (patched.size() > ELF64_E_FLAGS_OFFSET &&
                (patched[ELF64_E_FLAGS_OFFSET] & EF_AMDGPU_MACH_MASK) == machFrom)
            {
                patched[ELF64_E_FLAGS_OFFSET] =
                    (patched[ELF64_E_FLAGS_OFFSET] & ~EF_AMDGPU_MACH_MASK) | machTo;
            }

            for (std::size_t i = 0; i + from.size() <= patched.size(); ++i)
            {
                if (std::memcmp(&patched[i], from.data(), from.size()) == 0)
                {
                    std::memcpy(&patched[i], to.data(), to.size());
                }
            }

            return patched;
        }

#define CHECK_COMGR(call)                                                     \
    do {                                                                      \
        amd_comgr_status_t comgrStatus = (call);                              \
        if (comgrStatus != AMD_COMGR_STATUS_SUCCESS)                          \
        {                                                                     \
            cleanup();                                                        \
            return std::unexpected(make_error_code(MLSSErrorCode::ComgrError));\
        }                                                                     \
    } while (0)

        std::expected<std::vector<std::uint8_t>, std::error_code> linkRelocatableToExecutable(
            std::span<const std::uint8_t> relocatable,
            std::string_view targetIsa)
        {
            amd_comgr_data_t relocData{};
            amd_comgr_data_set_t relocSet{};
            amd_comgr_data_set_t execSet{};
            amd_comgr_action_info_t actionInfo{};
            amd_comgr_data_t execData{};

            auto cleanup = [&]()
            {
                amd_comgr_release_data(relocData);
                amd_comgr_release_data(execData);
                amd_comgr_destroy_data_set(relocSet);
                amd_comgr_destroy_data_set(execSet);
                amd_comgr_destroy_action_info(actionInfo);
            };

            CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &relocData));
            CHECK_COMGR(amd_comgr_set_data(relocData, relocatable.size(),
                        reinterpret_cast<const char*>(relocatable.data())));
            CHECK_COMGR(amd_comgr_set_data_name(relocData, "input.o"));

            CHECK_COMGR(amd_comgr_create_data_set(&relocSet));
            CHECK_COMGR(amd_comgr_data_set_add(relocSet, relocData));

            CHECK_COMGR(amd_comgr_create_action_info(&actionInfo));
            CHECK_COMGR(amd_comgr_action_info_set_isa_name(actionInfo, targetIsa.data()));

            CHECK_COMGR(amd_comgr_create_data_set(&execSet));
            CHECK_COMGR(amd_comgr_do_action(
                AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                actionInfo, relocSet, execSet));

            CHECK_COMGR(amd_comgr_action_data_get_data(execSet,
                        AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &execData));

            std::size_t execSize = 0;
            CHECK_COMGR(amd_comgr_get_data(execData, &execSize, nullptr));
            std::vector<std::uint8_t> result(execSize);
            CHECK_COMGR(amd_comgr_get_data(execData, &execSize, reinterpret_cast<char*>(result.data())));

            cleanup();
            return result;
        }

        // Parse a MsgPack uint value from buf at offset i. Returns false on failure.
        bool parseMsgpackUint32(const std::uint8_t* buf, std::size_t size, std::size_t& i, std::uint32_t& out)
        {
            if (i >= size) return false;
            std::uint8_t b = buf[i];
            if (b <= 0x7fu)          { out = b; i += 1; return true; }
            if (b == 0xccu && i + 1 < size) { out = buf[i+1]; i += 2; return true; }
            if (b == 0xcdu && i + 2 < size) { out = (static_cast<std::uint32_t>(buf[i+1]) << 8u) | buf[i+2]; i += 3; return true; }
            if (b == 0xceu && i + 4 < size) {
                out = (static_cast<std::uint32_t>(buf[i+1]) << 24u)
                    | (static_cast<std::uint32_t>(buf[i+2]) << 16u)
                    | (static_cast<std::uint32_t>(buf[i+3]) <<  8u)
                    |  static_cast<std::uint32_t>(buf[i+4]);
                i += 5; return true;
            }
            return false;
        }

        // Extract workgroup size by parsing the AMDGPU ELF .note section directly.
        // Reads .threadgroup_dimensions (PAL pipeline metadata) from SHT_NOTE sections.
        // Avoids invoking comgr, which causes STATUS_STACK_BUFFER_OVERRUN on Windows
        // when processing large Winograd relocatable ELFs.
        std::expected<MLSSdim3, std::error_code> extractWorkgroupSize(
            std::span<const std::uint8_t> binary)
        {
            constexpr std::size_t   kEhdrSize = 64u;
            constexpr std::size_t   kShdrSize = 64u;
            constexpr std::uint32_t SHT_NOTE  = 7u;

            const auto* raw = binary.data();
            const std::size_t sz = binary.size();

            if (sz < kEhdrSize || raw[0] != 0x7fu || raw[1] != 'E' || raw[2] != 'L' || raw[3] != 'F')
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
            }

            // Use memcpy for potentially unaligned ELF header fields
            std::uint64_t shoff{}; std::memcpy(&shoff, raw + 40u, 8u);
            std::uint16_t shentsize{}; std::memcpy(&shentsize, raw + 58u, 2u);
            std::uint16_t shnum{};    std::memcpy(&shnum,    raw + 60u, 2u);

            if (shoff == 0u || shentsize < kShdrSize || shoff + static_cast<std::uint64_t>(shnum) * shentsize > sz)
            {
                return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
            }

            static constexpr std::uint8_t kKey[] = ".threadgroup_dimensions";
            constexpr std::size_t kKeyLen = sizeof(kKey) - 1u;

            for (std::uint16_t si = 0u; si < shnum; ++si)
            {
                const auto* shdr = raw + shoff + static_cast<std::uint64_t>(si) * shentsize;
                std::uint32_t sh_type{};   std::memcpy(&sh_type,   shdr + 4u,  4u);
                std::uint64_t sh_offset{}; std::memcpy(&sh_offset, shdr + 24u, 8u);
                std::uint64_t sh_size{};   std::memcpy(&sh_size,   shdr + 32u, 8u);

                if (sh_type != SHT_NOTE || sh_size == 0u || sh_offset + sh_size > sz)
                {
                    continue;
                }

                const auto* note = raw + sh_offset;
                std::size_t noff = 0u;

                while (noff + 12u <= static_cast<std::size_t>(sh_size))
                {
                    std::uint32_t namesz{}; std::memcpy(&namesz, note + noff,      4u);
                    std::uint32_t descsz{}; std::memcpy(&descsz, note + noff + 4u, 4u);
                    noff += 12u;
                    noff += (namesz + 3u) & ~3u;
                    const auto* desc = note + noff;
                    const std::size_t desc_end = noff + descsz;
                    noff += (descsz + 3u) & ~3u;

                    if (noff > static_cast<std::size_t>(sh_size) || desc_end > static_cast<std::size_t>(sh_size))
                    {
                        break;
                    }

                    for (std::size_t ki = 0u; ki + kKeyLen <= descsz; ++ki)
                    {
                        if (std::memcmp(desc + ki, kKey, kKeyLen) != 0)
                        {
                            continue;
                        }

                        std::size_t vi = ki + kKeyLen;
                        if (vi >= descsz) break;

                        const std::uint8_t arrByte = desc[vi];
                        if ((arrByte & 0xf0u) != 0x90u) break;
                        if ((arrByte & 0x0fu) < 3u) break;
                        vi += 1u;

                        MLSSdim3 result{ 0x01u, 0x01u, 0x01u };
                        std::uint32_t* dims[3] = { &result.m_x, &result.m_y, &result.m_z };
                        bool ok = true;
                        for (std::uint32_t d = 0u; d < 3u; ++d)
                        {
                            if (!parseMsgpackUint32(desc, descsz, vi, *dims[d]))
                            {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) return result;
                    }
                }
            }

            return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
        }

#undef CHECK_COMGR



    } // anonymous namespace

    //=====================================================================================================================
    std::expected<std::string, std::error_code> getKernelName(const std::span<const std::uint8_t>& arr)
    {
        const auto invalidElf = [] {
            return std::unexpected(make_error_code(MLSSErrorCode::InvalidElfBinary));
        };

        if (arr.size() < sizeof(Elf64_Ehdr))
        {
            return invalidElf();
        }

        const std::uint8_t* const ptr = arr.data();
        const std::size_t size = arr.size();

        // Verify ELF magic number
        if (ptr[0] != 0x7f || ptr[1] != std::uint8_t{'E'} ||
            ptr[2] != std::uint8_t{'L'} || ptr[3] != std::uint8_t{'F'})
        {
            return invalidElf();
        }

        // Verify 64-bit ELF (EI_CLASS == ELFCLASS64)
        if (ptr[4] != 2)
        {
            return invalidElf();
        }

        const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(ptr);

        // Validate section header table offset
        if (ehdr->e_shoff == 0 || ehdr->e_shoff + ehdr->e_shnum * sizeof(Elf64_Shdr) > size)
        {
            return invalidElf();
        }

        const auto* shdrs = reinterpret_cast<const Elf64_Shdr*>(ptr + ehdr->e_shoff);

        // Find symbol table and its associated string table
        const Elf64_Shdr* symtab = nullptr;
        const Elf64_Shdr* strtab = nullptr;

        for (std::uint16_t i = 0; i < ehdr->e_shnum; ++i)
        {
            if (shdrs[i].sh_type == SHT_SYMTAB)
            {
                symtab = &shdrs[i];
                // sh_link points to the associated string table
                if (symtab->sh_link < ehdr->e_shnum)
                {
                    strtab = &shdrs[symtab->sh_link];
                }
                break;
            }
        }

        if (symtab == nullptr || strtab == nullptr)
        {
            return invalidElf();
        }

        // Validate symbol table and string table offsets
        if (symtab->sh_offset + symtab->sh_size > size ||
            strtab->sh_offset + strtab->sh_size > size)
        {
            return invalidElf();
        }

        const auto* symbols = reinterpret_cast<const Elf64_Sym*>(ptr + symtab->sh_offset);
        const char* strings = reinterpret_cast<const char*>(ptr + strtab->sh_offset);
        const std::size_t numSymbols = symtab->sh_size / sizeof(Elf64_Sym);

        std::vector<std::string> kernelNames;

        // First pass: collect kernel descriptor symbols (*.kd) — the canonical AMDGPU identifiers
        for (std::size_t i = 0; i < numSymbols; ++i)
        {
            if (symbols[i].st_name >= strtab->sh_size)
            {
                continue;
            }

            std::string name(strings + symbols[i].st_name);

            if (endsWith(name, ".kd"))
            {
                kernelNames.push_back(name.substr(0, name.size() - 3));
            }
        }

        // Fallback: if no .kd descriptors, look for mangled FUNC symbols
        if (kernelNames.empty())
        {
            for (std::size_t i = 0; i < numSymbols; ++i)
            {
                const auto& sym = symbols[i];

                if (sym.st_name >= strtab->sh_size)
                {
                    continue;
                }

                std::string name(strings + sym.st_name);

                // Accept C++-mangled HIP kernels (_Z...) and
                // AMDGPU compute-shader entry points (_amdgpu_cs_main).
                if (ELF64_ST_TYPE(sym.st_info) == STT_FUNC &&
                    sym.st_shndx != 0 &&
                    ((name.size() > 2 && name[0] == '_' && name[1] == 'Z') ||
                     name == "_amdgpu_cs_main"))
                {
                    kernelNames.push_back(name);
                }
            }
        }

        if (kernelNames.size() != 1u)
        {
            return invalidElf();
        }

        return kernelNames[0];
    }


    std::expected<std::vector<std::uint8_t>, std::error_code> patchGfxInElf(
        std::span<const std::uint8_t> input,
        const GfxIpTriple& gfxIpHighEnd, const GfxIpTriple& gfxIpTarget)
    {
        if (!areGfxIpsCompatible(gfxIpHighEnd, gfxIpTarget))
        {
            return std::unexpected(make_error_code(MLSSErrorCode::ArchitectureNotSupported));
        }

        if (gfxIpHighEnd == gfxIpTarget)
        {
            return std::vector<std::uint8_t>(input.begin(), input.end());
        }

        auto elfShaderQuery = gfxIpTripleToElfMatch(gfxIpHighEnd);
        if (!elfShaderQuery.has_value())
        {
            return std::unexpected(elfShaderQuery.error());
        }

        auto elfTargetQuery = gfxIpTripleToElfMatch(gfxIpTarget);
        if (!elfTargetQuery.has_value())
        {
            return std::unexpected(elfTargetQuery.error());
        }

        auto shaderIsaResult = gfxIpToIsaString(gfxIpHighEnd);
        if (!shaderIsaResult.has_value())
        {
            return std::unexpected(shaderIsaResult.error());
        }

        auto targetIsaResult = gfxIpToIsaString(gfxIpTarget);
        if (!targetIsaResult.has_value())
        {
            return std::unexpected(targetIsaResult.error());
        }

        return patchGfxVersion(input,
            shaderIsaResult.value(), elfShaderQuery.value(),
            targetIsaResult.value(), elfTargetQuery.value());
    }

    std::expected<std::vector<std::uint8_t>, std::error_code> getNonRelocatable(const std::span<const std::uint8_t>& arr,
        const GfxIpTriple& gfxIpHighEnd, const GfxIpTriple& gfxIpTarget)
    {
        auto patchedQuery = patchGfxInElf(arr, gfxIpHighEnd, gfxIpTarget);
        if (!patchedQuery.has_value())
        {
            return std::unexpected(patchedQuery.error());
        }

        auto targetIsaResult = gfxIpToIsaString(gfxIpTarget);
        if (!targetIsaResult.has_value())
        {
            return std::unexpected(targetIsaResult.error());
        }
        const std::string comgrIsa = "amdgcn-amd-amdhsa--" + targetIsaResult.value();

        auto executableQuery = linkRelocatableToExecutable(patchedQuery.value(), comgrIsa);
        if (!executableQuery.has_value())
        {
            return std::unexpected(executableQuery.error());
        }

        return std::move(executableQuery.value());
    }

    //=====================================================================================================================
    std::expected<MLSSdim3, std::error_code> getWorkgroupSize(const std::span<const std::uint8_t>& arr)
    {
        return extractWorkgroupSize(arr);
    }

} // namespace mlss
