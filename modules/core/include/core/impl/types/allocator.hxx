#pragma once

#include <cstddef>
#include <bit>

namespace mlss
{
    //=================================================================================================================
    //                                   CacheAlignedAllocator                                                                   
    //=================================================================================================================
    template<typename T, std::size_t cache_line_size_ = std::hardware_constructive_interference_size>
    class CacheAlignedAllocator
    {
    public:

        static constexpr std::size_t CACHE_LINE_SIZE = cache_line_size_;

        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using const_pointer = const T*;

        //---------------------------------------------------------------------
        template<typename U>
        struct rebind
        {
            using other = CacheAlignedAllocator<U>;
        };

        //---------------------------------------------------------------------
        constexpr CacheAlignedAllocator() noexcept = default;

        //---------------------------------------------------------------------
        template<typename U>
        constexpr inline CacheAlignedAllocator(const CacheAlignedAllocator<U>&) noexcept {}

        //---------------------------------------------------------------------
        [[nodiscard]] pointer allocate(std::size_t n);

        //---------------------------------------------------------------------
        void deallocate(pointer ptr, std::size_t) noexcept;

        //---------------------------------------------------------------------
        template<typename U>
        constexpr inline bool operator==(const CacheAlignedAllocator<U>&) const noexcept
        {
            return true;
        }

        //---------------------------------------------------------------------
        template<typename U>
        constexpr inline bool operator!=(const CacheAlignedAllocator<U>&) const noexcept
        {
            return false;
        }
    };

} // mlss
