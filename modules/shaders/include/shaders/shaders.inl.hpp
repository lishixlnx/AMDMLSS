/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

namespace mlss
{

namespace shaders
{

//=====================================================================================================================
template<std::ranges::contiguous_range T, std::size_t M>
requires BinaryType<T>
ShaderType<T> make_shader_type(
    std::span<const std::ranges::range_value_t<T>, M> binary,
    std::string_view              kernelName,
    std::string_view              compilerVersion,
    std::uint32_t                 codeObjectVersion,
    bool                          isRelocatable,
    ShaderTypesFlags              shaderType)
{
    ShaderType<T> result;
    using Elem = std::ranges::range_value_t<T>;

    if constexpr (requires(T& t, std::size_t n) { t.resize(n); })
    {
        result.m_binary.resize(binary.size());
        std::ranges::copy(binary, std::ranges::begin(result.m_binary));
    }
    else if constexpr (std::is_constructible_v<T, const Elem*, std::size_t>)
    {
        result.m_binary = T(binary.data(), binary.size());
    }
    else
    {
        std::ranges::copy(binary, std::ranges::begin(result.m_binary));
    }

    result.m_kernelName        = typename ShaderType<T>::string_type(
        kernelName.empty()
            ? mlss::getKernelName(
                  reinterpret_cast<const std::byte*>(binary.data()),
                  binary.size())
            : kernelName);
    result.m_compilerVersion   = compilerVersion;
    result.m_codeObjectVersion = codeObjectVersion;
    result.m_isRelocatable     = isRelocatable;
    result.m_shaderType        = shaderType;
    return result;
}

//=====================================================================================================================
template<std::size_t M, BinaryElement E>
StaticShaderType<M> make_static_shader_type(
    std::span<const E, M>         binary,
    std::string_view              kernelName,
    std::string_view              compilerVersion,
    std::uint32_t                 codeObjectVersion,
    bool                          isRelocatable,
    ShaderTypesFlags              shaderType)
{
    if constexpr (std::same_as<E, std::uint8_t>)
    {
        return make_shader_type<std::array<std::uint8_t, M>>(
            binary,
            kernelName,
            compilerVersion,
            codeObjectVersion,
            isRelocatable,
            shaderType);
    }
    else
    {
        return make_shader_type<std::array<std::uint8_t, M>>(
            std::span<const std::uint8_t, M>(
                reinterpret_cast<const std::uint8_t*>(binary.data()),
                binary.size()),
            kernelName,
            compilerVersion,
            codeObjectVersion,
            isRelocatable,
            shaderType);
    }
}

//=====================================================================================================================
inline DynamicShaderType make_dynamic_shader_type(
    std::span<const std::byte> binary,
    std::string_view           kernelName,
    std::string_view           compilerVersion,
    std::uint32_t              codeObjectVersion,
    bool                       isRelocatable,
    ShaderTypesFlags           shaderType)
{
    return make_shader_type<std::vector<std::uint8_t, mlss::CacheAlignedAllocator<std::uint8_t>>>(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(binary.data()),
            binary.size()),
        kernelName,
        compilerVersion,
        codeObjectVersion,
        isRelocatable,
        shaderType);
}

//=====================================================================================================================
template<BinaryElement E>
ShaderDescriptor make_shader_descriptor(
    std::span<const E>            binary,
    std::string_view              kernelName,
    std::string_view              compilerVersion,
    std::uint32_t                 codeObjectVersion,
    bool                          isRelocatable,
    ShaderTypesFlags              shaderType)
{
    return make_shader_type<std::span<const std::uint8_t>>(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(binary.data()),
            binary.size()),
        kernelName,
        compilerVersion,
        codeObjectVersion,
        isRelocatable,
        shaderType);
}

//=====================================================================================================================
template<std::ranges::contiguous_range T>
requires BinaryType<T>
ShaderDescriptor make_shader_descriptor(const ShaderType<T>& shader)
{
    return ShaderDescriptor{
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(std::ranges::data(shader.m_binary)),
            std::ranges::size(shader.m_binary)),
        typename ShaderDescriptor::string_type(shader.m_kernelName),
        shader.m_compilerVersion,
        shader.m_codeObjectVersion,
        shader.m_isRelocatable,
        shader.m_shaderType
    };
}

//=====================================================================================================================
template<std::ranges::contiguous_range T>
requires BinaryType<T>
std::unique_ptr<Binaries::Blob> make_binary_blob(const ShaderType<T>& shader)
{
    std::string name(shader.m_kernelName);
    return std::make_unique<Binaries::Blob>(Binaries::Blob{
        shader.m_binary.data(),
        shader.m_binary.size(),
        MLSS_BINARY_TYPE_ELF,
        0,
        name
    });
}

//=====================================================================================================================
template<typename T>
requires (!is_library_shader_type<std::remove_cvref_t<T>>::value)
      && requires(const T& t) {
             { t.m_binary.data() } -> std::convertible_to<const void*>;
             { t.m_binary.size() } -> std::convertible_to<std::size_t>;
             { t.m_kernelName }    -> std::convertible_to<std::string_view>;
         }
std::unique_ptr<Binaries::Blob> make_binary_blob(const T& shader)
{
    std::string name(shader.m_kernelName);
    return std::make_unique<Binaries::Blob>(Binaries::Blob{
        shader.m_binary.data(),
        shader.m_binary.size(),
        MLSS_BINARY_TYPE_ELF,
        0,
        name
    });
}

} // shaders

} // mlss
