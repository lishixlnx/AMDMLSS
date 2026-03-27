/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/impl/utils/elf.hxx"

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
        constexpr std::uint32_t SHT_SYMTAB = 2;
        constexpr std::uint32_t SHT_STRTAB = 3;
        constexpr std::uint8_t STT_FUNC = 2;
        constexpr std::uint8_t STT_OBJECT = 1;

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

    } // anonymous namespace

    //=====================================================================================================================
    std::string getKernelName(const std::byte* const ptr, const std::size_t size)
    {
        if (ptr == nullptr || size < sizeof(Elf64_Ehdr))
        {
            throw std::runtime_error("Invalid ELF binary: null pointer or insufficient size");
        }

        // Verify ELF magic number
        if (ptr[0] != std::byte{0x7f} || ptr[1] != std::byte{'E'} ||
            ptr[2] != std::byte{'L'} || ptr[3] != std::byte{'F'})
        {
            throw std::runtime_error("Invalid ELF binary: bad magic number");
        }

        // Verify 64-bit ELF
        if (ptr[4] != std::byte{2})
        {
            throw std::runtime_error("Invalid ELF binary: not a 64-bit ELF");
        }

        const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(ptr);

        // Validate section header table offset
        if (ehdr->e_shoff == 0 || ehdr->e_shoff + ehdr->e_shnum * sizeof(Elf64_Shdr) > size)
        {
            throw std::runtime_error("Invalid ELF binary: invalid section header table");
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
            throw std::runtime_error("Invalid ELF binary: no symbol table found");
        }

        // Validate symbol table and string table offsets
        if (symtab->sh_offset + symtab->sh_size > size ||
            strtab->sh_offset + strtab->sh_size > size)
        {
            throw std::runtime_error("Invalid ELF binary: symbol/string table out of bounds");
        }

        const auto* symbols = reinterpret_cast<const Elf64_Sym*>(ptr + symtab->sh_offset);
        const char* strings = reinterpret_cast<const char*>(ptr + strtab->sh_offset);
        const std::size_t numSymbols = symtab->sh_size / sizeof(Elf64_Sym);

        std::vector<std::string> kernelNames;

        // Search for kernel symbols
        // AMD GPU kernels have kernel descriptors with ".kd" suffix in the symbol table
        for (std::size_t i = 0; i < numSymbols; ++i)
        {
            const auto& sym = symbols[i];

            // Skip if name index is out of bounds
            if (sym.st_name >= strtab->sh_size)
            {
                continue;
            }

            const char* symName = strings + sym.st_name;
            std::string name(symName);

            // AMD GPU kernel descriptors end with ".kd"
            if (endsWith(name, ".kd"))
            {
                // Remove the ".kd" suffix to get the actual kernel name
                kernelNames.push_back(name.substr(0, name.size() - 3));
            }
            // Also check for FUNC symbols that might be kernels (without .kd suffix)
            else if (ELF64_ST_TYPE(sym.st_info) == STT_FUNC &&
                     sym.st_shndx != 0 &&
                     !name.empty() &&
                     name[0] != '.') // Skip section names
            {
                // Check if this looks like a mangled kernel name (starts with _Z for C++ mangling)
                if (name.size() > 2 && name[0] == '_' && name[1] == 'Z')
                {
                    // Only add if we haven't already found a .kd version
                    bool alreadyFound = false;
                    for (const auto& kn : kernelNames)
                    {
                        if (kn == name)
                        {
                            alreadyFound = true;
                            break;
                        }
                    }
                    if (!alreadyFound)
                    {
                        kernelNames.push_back(name);
                    }
                }
            }
        }

        if (kernelNames.empty())
        {
            throw std::runtime_error("No kernel found in ELF binary");
        }

        if (kernelNames.size() > 1)
        {
            throw std::runtime_error("Multiple kernels found in ELF binary (expected single kernel)");
        }

        return kernelNames[0];
    }

} // namespace mlss
