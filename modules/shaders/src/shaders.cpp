#include "shaders/shaders.hpp"

namespace mlss
{

    //=====================================================================================================================
    std::unique_ptr<Binaries::Blob> make_binary_blob(const ShaderDescriptor& shader)
    {
        std::string name(shader.m_kernelName);
        if (name.empty())
        {
            try
            {
                name = getKernelName(std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(shader.m_binary.data()),
                    shader.m_binary.size()));
            }
            catch (const std::runtime_error&) {}
        }
        return std::make_unique<Binaries::Blob>(Binaries::Blob{
            shader.m_binary.data(),
            shader.m_binary.size(),
            static_cast<std::uint32_t>(BinaryTypeFlags::ELF),
            0,
            name});
    }

    //=====================================================================================================================
    std::unique_ptr<Binaries::Blob> make_binary_blob(const ShaderDescriptorByte& shader)
    {
        std::string name(shader.m_kernelName);
        if (name.empty())
        {
            try
            {
                name = getKernelName(std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(shader.m_binary.data()),
                    shader.m_binary.size()));
            }
            catch (const std::runtime_error&) {}
        }
        return std::make_unique<Binaries::Blob>(Binaries::Blob{
            shader.m_binary.data(),
            shader.m_binary.size(),
            static_cast<std::uint32_t>(BinaryTypeFlags::ELF),
            0,
            name});
    }

} // namespace mlss
