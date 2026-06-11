/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    using size_t = std::size_t;

    //=====================================================================================================================
    //                                   CacheAlignedAllocator
    //=====================================================================================================================

    //---------------------------------------------------------------------
    template <typename T, size_t cache_line_size_>
    typename CacheAlignedAllocator<T, cache_line_size_>::pointer CacheAlignedAllocator<T, cache_line_size_>::allocate(std::size_t n)
    {
        if (n == 0)
        {
            return nullptr;
        }

        size_t bytes = n * sizeof(T);
        void* ptr = nullptr;

#ifdef _MSC_VER
        ptr = _aligned_malloc(bytes, CACHE_LINE_SIZE);
#else
        // Round up size to a multiple of the alignment for platforms that require it
        std::size_t aligned_size = (bytes + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);

#if defined(__ANDROID__) || (defined(__APPLE__) && defined(__MACH__))
        ptr = std::aligned_alloc(CACHE_LINE_SIZE, aligned_size);
#else
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size) != 0)
        {
            ptr = nullptr;
        }
#endif
#endif
        if (!ptr)
        {
            throw std::bad_alloc();
        }

        return static_cast<T*>(ptr);
    }

    //---------------------------------------------------------------------
    template <typename T, std::size_t cache_line_size_>
    void CacheAlignedAllocator<T, cache_line_size_>::deallocate(pointer ptr, std::size_t) noexcept
    {
        if (ptr)
        {
#ifdef _MSC_VER
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }

} // namespace mlss
