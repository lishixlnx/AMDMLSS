/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "core/core.hpp"

namespace mlss
{
    //=====================================================================================================================
    //                                   Any
    //=====================================================================================================================

    //---------------------------------------------------------------------
    void* Any::allocateStorage(size_t size, size_t alignment)
    {
        if (size == 0)
        {
            return nullptr;
        }

        // Calculate padding needed for alignment
        size_t padding = alignment > 1 ? alignment - 1 : 0;
        size_t totalSize = size + padding;

        // Resize the storage vector
        m_storage.resize(totalSize);

        // Get aligned pointer
        void* basePtr = m_storage.data();
        uintptr_t addr = reinterpret_cast<uintptr_t>(basePtr);
        uintptr_t alignedAddr = (addr + alignment - 1) & ~(alignment - 1);

        return reinterpret_cast<void*>(alignedAddr);
    }

    //---------------------------------------------------------------------
    void Any::reset()
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (!m_storage.empty() && m_typeInfo)
        {
            void* objPtr = getStoragePtr();
            if (objPtr)
            {
                m_typeInfo->destroy(objPtr);
            }
        }

        m_storage.clear();
        m_storage.shrink_to_fit();

        // Smart pointer automatically handles cleanup
        m_typeInfo.reset();
    }

    //---------------------------------------------------------------------
    void* Any::getStoragePtr() const noexcept
    {
        if (m_storage.empty() || !m_typeInfo)
        {
            return nullptr;
        }

        void* basePtr = const_cast<std::uint8_t*>(m_storage.data());
        uintptr_t addr = reinterpret_cast<uintptr_t>(basePtr);
        uintptr_t alignedAddr = (addr + m_typeInfo->alignment - 1) & ~(m_typeInfo->alignment - 1);

        return reinterpret_cast<void*>(alignedAddr);
    }

    //---------------------------------------------------------------------
    Any::Any(const Any& other)
    {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        if (other.m_typeInfo && !other.m_storage.empty())
        {
            // Always make a deep copy of the TypeInfo
            m_typeInfo = std::make_unique<TypeInfo>(*other.m_typeInfo);

            void* newStorage = allocateStorage(m_typeInfo->size, m_typeInfo->alignment);
            const void* otherStorage = other.getStoragePtr();

            if (newStorage && otherStorage)
            {
                m_typeInfo->copyConstruct(newStorage, otherStorage);
            }
        }
    }

    //---------------------------------------------------------------------
    Any::Any(Any&& other) noexcept
    {
        std::lock_guard<std::mutex> lock(other.m_mutex);
        m_storage = std::move(other.m_storage);
        m_typeInfo = std::move(other.m_typeInfo);
    }

    //---------------------------------------------------------------------
    Any::~Any()
    {
        reset();
    }

    //---------------------------------------------------------------------
    Any& Any::operator=(const Any& other)
    {
        if (this != &other)
        {
            Any temp(other);
            swap(temp);
        }
        return *this;
    }

    //---------------------------------------------------------------------
    Any& Any::operator=(Any&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            std::scoped_lock lock(m_mutex, other.m_mutex);

            m_storage = std::move(other.m_storage);
            m_typeInfo = std::move(other.m_typeInfo);
        }
        return *this;
    }

    //---------------------------------------------------------------------
    void Any::swap(Any& other) noexcept
    {
        if (this != &other)
        {
            std::scoped_lock lock(m_mutex, other.m_mutex);
            std::swap(m_storage, other.m_storage);
            std::swap(m_typeInfo, other.m_typeInfo);
        }
    }

    //---------------------------------------------------------------------
    bool Any::hasValue() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return !m_storage.empty() && m_typeInfo != nullptr;
    }

    //---------------------------------------------------------------------
    bool Any::isRange() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_typeInfo ? m_typeInfo->isRange : false;
    }

    size_t Any::size() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_typeInfo ? m_typeInfo->size : 0;
    }

    //---------------------------------------------------------------------
    size_t Any::storageSize() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_storage.size();
    }

    //---------------------------------------------------------------------
    bool Any::isCacheAligned() const noexcept
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_storage.empty())
        {
            return false;
        }

        uintptr_t addr = reinterpret_cast<uintptr_t>(m_storage.data());
        return (addr % CacheAlignedStorage::allocator_type::CACHE_LINE_SIZE) == 0;
    }

} // namespace mlss
