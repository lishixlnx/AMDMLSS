/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    template<typename T>
    constexpr std::uint32_t getMLSSTypeEnum()
    {
        if constexpr (std::is_same_v<T, bool>) 
        {
            return MLSS_BOOL;
        }
        else if constexpr (std::is_same_v<T, std::int8_t>) 
        {
            return MLSS_INT8;
        }
        else if constexpr (std::is_same_v<T, std::uint8_t>) 
        {
            return MLSS_UINT8;
        }
        else if constexpr (std::is_same_v<T, std::int16_t>) 
        {
            return MLSS_INT16;
        }
        else if constexpr (std::is_same_v<T, std::uint16_t>) 
        {
            return MLSS_UINT16;
        }
        else if constexpr (std::is_same_v<T, std::int32_t>) 
        {
            return MLSS_INT32;
        }
        else if constexpr (std::is_same_v<T, std::uint32_t>) 
        {
            return MLSS_UINT32;
        }
        else if constexpr (std::is_same_v<T, std::int64_t>) 
        {
            return MLSS_INT64;
        }
        else if constexpr (std::is_same_v<T, std::uint64_t>) 
        {
            return MLSS_UINT64;
        }
        else if constexpr (std::is_same_v<T, float>) 
        {
            return MLSS_FLOAT32;
        }
        else if constexpr (std::is_same_v<T, double>) 
        {
            return MLSS_FLOAT64;
        }
        else if constexpr (std::is_same_v<T, MLSSarg>)
        {
            return MLSS_ARG;
        }
        else if constexpr (std::is_same_v<T, MLSSdim3>)
        {
            return MLSS_DIM3;
        }
        else if constexpr (std::is_same_v<T, Context>)
        {
            return MLSS_CONTEXT;
        }
        else if constexpr (std::is_same_v<T, Binaries>)
        {
            return MLSS_BINARY;
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            return MLSS_STRING;
        }
        else
        {
            return MLSS_UNKNOWN_TYPE;
        }
    }

    //=====================================================================================================================
    template<typename T>
    MLSSvector createTypedVector(const T* data, size_t count)
    {
        MLSSvector vec;
        vec.m_size = count;
        vec.m_type = getMLSSTypeEnum<T>();
        vec.m_handle = 0;

        if (data && count > 0)
        {
            // Create std::vector<T> and wrap in Any
            std::vector<T> storage(data, data + count);
            Any any_obj = std::move(storage);

            // Store in MemoryManager
            vec.m_handle = MemoryManager::addObject(std::move(any_obj));

            // Mark as initialized
            Any* new_any = MemoryManager::template getPointer<Any>(vec.m_handle);
            if (new_any)
            {
                MemoryManager::markAsInitialized(&new_any);
            }
        }

        return vec;
    }

    //=====================================================================================================================
    template<typename T>
    const T* getVectorData(const MLSSvector& vec)
    {
        if (vec.m_handle == 0)
        {
            return nullptr;
        }

        Any* any_obj = MemoryManager::template getPointer<Any>(vec.m_handle);
        if (any_obj && anyIs<std::vector<T>>(*any_obj))
        {
            const auto& storage = anyCast<const std::vector<T>&>(*any_obj);
            return storage.data();
        }

        return nullptr;
    }

    //=====================================================================================================================
    template<typename T>
    T* getVectorDataMutable(const MLSSvector& vec)
    {
        if (vec.m_handle == 0)
        {
            return nullptr;
        }

        Any* any_obj = MemoryManager::template getPointer<Any>(vec.m_handle);
        if (any_obj && anyIs<std::vector<T>>(*any_obj))
        {
            auto& storage = anyCast<std::vector<T>&>(*any_obj);
            return storage.data();
        }

        return nullptr;
    }

} // namespace mlss

