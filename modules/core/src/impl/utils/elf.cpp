/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

#include "core/core.hpp"

#include <amd_comgr/amd_comgr.h>
#include <charconv>
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

        std::expected<MLSSdim3, std::error_code> extractWorkgroupSize(
            std::span<const std::uint8_t> binary)
        {
            amd_comgr_data_t data{};
            amd_comgr_metadata_node_t rootMeta{};
            amd_comgr_metadata_node_t kernelsMeta{};
            amd_comgr_metadata_node_t kernel0Meta{};
            amd_comgr_metadata_node_t wgsMeta{};
            amd_comgr_metadata_node_t dimMeta[0x03]{};
            bool hasRoot    = false;
            bool hasKernels = false;
            bool hasKernel0 = false;
            bool hasWgs     = false;
            std::uint32_t dimCount = 0x00u;

            auto cleanup = [&]()
            {
                for (std::uint32_t i = 0x00u; i < dimCount; ++i)
                {
                    amd_comgr_destroy_metadata(dimMeta[i]);
                }
                if (hasWgs)     amd_comgr_destroy_metadata(wgsMeta);
                if (hasKernel0) amd_comgr_destroy_metadata(kernel0Meta);
                if (hasKernels) amd_comgr_destroy_metadata(kernelsMeta);
                if (hasRoot)    amd_comgr_destroy_metadata(rootMeta);
                amd_comgr_release_data(data);
            };

            CHECK_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &data));
            CHECK_COMGR(amd_comgr_set_data(data, binary.size(),
                        reinterpret_cast<const char*>(binary.data())));

            CHECK_COMGR(amd_comgr_get_data_metadata(data, &rootMeta));
            hasRoot = true;

            CHECK_COMGR(amd_comgr_metadata_lookup(rootMeta, "amdhsa.kernels", &kernelsMeta));
            hasKernels = true;

            CHECK_COMGR(amd_comgr_index_list_metadata(kernelsMeta, 0, &kernel0Meta));
            hasKernel0 = true;

            auto parseMetadataUint32 = [](amd_comgr_metadata_node_t node)
                -> std::expected<std::uint32_t, std::error_code>
            {
                std::size_t strSize = 0x00u;
                if (amd_comgr_get_metadata_string(node, &strSize, nullptr) != AMD_COMGR_STATUS_SUCCESS)
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }

                std::string buf(strSize, '\0');
                if (amd_comgr_get_metadata_string(node, &strSize, buf.data()) != AMD_COMGR_STATUS_SUCCESS)
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }

                std::uint32_t val = 0x00u;
                auto [ptr, ec] = std::from_chars(buf.data(), buf.data() + buf.size(), val);
                if (ec != std::errc{})
                {
                    return std::unexpected(make_error_code(MLSSErrorCode::ShaderInvalidParameters));
                }
                return val;
            };

            amd_comgr_status_t wgsStatus = amd_comgr_metadata_lookup(
                kernel0Meta, ".reqd_workgroup_size", &wgsMeta);

            if (wgsStatus == AMD_COMGR_STATUS_SUCCESS)
            {
                hasWgs = true;
                MLSSdim3 result{ 0x01u, 0x01u, 0x01u };

                for (std::uint32_t i = 0x00u; i < 0x03u; ++i)
                {
                    CHECK_COMGR(amd_comgr_index_list_metadata(wgsMeta, i, &dimMeta[i]));
                    dimCount = i + 0x01u;

                    auto valResult = parseMetadataUint32(dimMeta[i]);
                    if (!valResult.has_value())
                    {
                        cleanup();
                        return std::unexpected(valResult.error());
                    }

                    if (i == 0x00u) result.m_x = valResult.value();
                    if (i == 0x01u) result.m_y = valResult.value();
                    if (i == 0x02u) result.m_z = valResult.value();
                }

                cleanup();
                return result;
            }

            wgsStatus = amd_comgr_metadata_lookup(
                kernel0Meta, ".max_flat_workgroup_size", &wgsMeta);

            if (wgsStatus == AMD_COMGR_STATUS_SUCCESS)
            {
                hasWgs = true;

                auto valResult = parseMetadataUint32(wgsMeta);
                if (!valResult.has_value())
                {
                    cleanup();
                    return std::unexpected(valResult.error());
                }

                cleanup();
                return MLSSdim3{ valResult.value(), 0x01u, 0x01u };
            }

            cleanup();
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

                if (ELF64_ST_TYPE(sym.st_info) == STT_FUNC &&
                    sym.st_shndx != 0 &&
                    name.size() > 2 && name[0] == '_' && name[1] == 'Z')
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
