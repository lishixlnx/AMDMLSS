/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <span>
#include "core/core.hpp"

#define MLSS_ASSERT_ALWAYS() assert(false);
#define MLSS_ASSERT(cdt) assert(cdt);

#if 0
#define MLSS_MAKE_BLOB(var) mlss::makeBinaryBlob(var, #var)
#endif

namespace mlss
{

        template <typename T, typename... U>
        concept either = (std::same_as<T, U> || ...);

        template <typename T>
        concept BinaryElement = std::same_as<T, std::byte> || std::same_as<T, std::uint8_t>;

        template <typename T>
        concept ShaderElement = BinaryElement<T> || std::same_as<T, std::uint32_t>;

        template <typename R>
        concept BinaryType = std::ranges::range<R> && either<std::ranges::range_value_t<R>, std::byte, std::uint8_t, std::uint32_t>;

        template <typename T>
        using CacheAlignedVector = std::vector<T, mlss::CacheAlignedAllocator<T>>;

        template <typename R>
        concept BinaryTypeRange = std::ranges::contiguous_range<R> && BinaryElement<std::ranges::range_value_t<R>>;

        template <typename R>
        concept ShaderStorageRange = std::ranges::contiguous_range<R> && ShaderElement<std::ranges::range_value_t<R>>;

        template <typename T>
        inline constexpr bool is_shader_range_v = false;

        template <ShaderElement B, std::size_t N>
        inline constexpr bool is_shader_range_v<std::array<B, N>> = true;

        template <ShaderElement B>
        inline constexpr bool is_shader_range_v<std::span<const B>> = true;

        template <ShaderElement B>
        inline constexpr bool is_shader_range_v<CacheAlignedVector<B>> = true;

        /// Contiguous shader storage: std::array, spans, or cache-aligned vectors of byte/uint8_t/uint32_t (see is_shader_range_v).
        template <class T>
        concept ShaderTypeRange = is_shader_range_v<T>;

        //=====================================================================================================================
        template <ShaderStorageRange T>
        struct ShaderType
        {
#if MLSS_ENABLE_SHADER_DESCRIPTOR_ONLY
            using string_type = std::string_view;
#else
            using string_type = std::string;
#endif

            using element_type = std::ranges::range_value_t<T>;
            using storage_type = T;

            storage_type m_binary = {};
            string_type m_kernelName = {};
            std::string_view m_compilerVersion = {};
            std::uint32_t m_codeObjectVersion = 0;
            bool m_isRelocatable = false;
            ShaderTypesFlags m_shaderType = ShaderTypesFlags::UNKNOWN;
        };

        template <std::size_t M>
        using StaticShaderType = ShaderType<std::array<std::uint8_t, M>>;

        template <std::size_t M>
        using StaticShaderByteType = ShaderType<std::array<std::byte, M>>;

        template <std::size_t M>
        using StaticShaderILType = ShaderType<std::array<std::uint32_t, M>>;

        using DynamicShaderType = ShaderType<std::vector<std::uint8_t, mlss::CacheAlignedAllocator<std::uint8_t>>>;

        using DynamicShaderByteType = ShaderType<std::vector<std::byte, mlss::CacheAlignedAllocator<std::byte>>>;

        using DynamicShaderILType = ShaderType<std::vector<std::uint32_t, mlss::CacheAlignedAllocator<std::uint32_t>>>;
        //=====================================================================================================================
        /// A non-owning view over binary shader data, borrowing from StaticShaderType or DynamicShaderType.
        using ShaderDescriptorType = ShaderType<std::span<const std::uint8_t>>;

        using ShaderDescriptorByteType = ShaderType<std::span<const std::byte>>;

        using ShaderDescriptor = ShaderDescriptorType;
        using ShaderDescriptorByte = ShaderDescriptorByteType;

        //=====================================================================================================================
        template <std::ranges::contiguous_range T, std::size_t M = std::dynamic_extent>
            requires BinaryType<T>
        ShaderType<T> make_shader_type(
            std::span<const std::ranges::range_value_t<T>, M> binary,
            std::string_view kernelName,
            std::string_view compilerVersion,
            std::uint32_t codeObjectVersion,
            bool isRelocatable,
            ShaderTypesFlags shaderType);

        //=====================================================================================================================
        /// Fixed-size shader blob; `std::byte` and `std::uint8_t` element spans share one implementation.
        template <std::size_t M, BinaryElement E>
        StaticShaderType<M> make_static_shader_type(
            std::span<const E, M> binary,
            std::string_view kernelName,
            std::string_view compilerVersion,
            std::uint32_t codeObjectVersion,
            bool isRelocatable,
            ShaderTypesFlags shaderType);

        //=====================================================================================================================
        /// Runtime-sized shader blob; defined in shaders.inl.hpp.
        DynamicShaderType make_dynamic_shader_type(
            std::span<const std::byte> binary,
            std::string_view kernelName,
            std::string_view compilerVersion,
            std::uint32_t codeObjectVersion,
            bool isRelocatable,
            ShaderTypesFlags shaderType);

        //=====================================================================================================================
        /// Non-owning view from raw bytes (`std::byte` or `std::uint8_t`); implementation in shaders.inl.hpp.
        template <BinaryElement E>
        ShaderDescriptor make_shader_descriptor(
            std::span<const E> binary,
            std::string_view kernelName,
            std::string_view compilerVersion,
            std::uint32_t codeObjectVersion,
            bool isRelocatable,
            ShaderTypesFlags shaderType);

        //=====================================================================================================================
        /// Non-owning view from any `ShaderType` storage (element layout matches uint8_t for ELF).
        template <std::ranges::contiguous_range T>
            requires BinaryType<T>
        ShaderDescriptor make_shader_descriptor(const ShaderType<T>& shader);

        //=====================================================================================================================
        template <std::ranges::contiguous_range T>
            requires BinaryType<T>
        std::unique_ptr<Binaries::Blob> make_binary_blob(const ShaderType<T>& shader);

        //=====================================================================================================================
        std::unique_ptr<Binaries::Blob> make_binary_blob(const ShaderDescriptor& shader);

        std::unique_ptr<Binaries::Blob> make_binary_blob(const ShaderDescriptorByte& shader);

        //=====================================================================================================================
        template <typename>
        struct is_library_shader_type : std::false_type
        {
        };
        template <ShaderStorageRange R>
        struct is_library_shader_type<ShaderType<R>> : std::true_type
        {
        };

        /// Generic overload for external shader types (e.g. C++20 module ShaderType<N>) that expose
        /// m_binary (with .data()/.size()) and m_kernelName but aren't mlss::ShaderType<T>.
        template <typename T>
            requires(!is_library_shader_type<std::remove_cvref_t<T>>::value) && requires(const T& t) {
                { t.m_binary.data() } -> std::convertible_to<const void*>;
                { t.m_binary.size() } -> std::convertible_to<std::size_t>;
                { t.m_kernelName } -> std::convertible_to<std::string_view>;
            }
        std::unique_ptr<Binaries::Blob> make_binary_blob(const T& shader);

} // namespace mlss

#include "shaders.inl.hpp"
