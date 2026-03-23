#include "core/core.hpp"

namespace mlss
{












    //=====================================================================================================================
    // MLSSvector Function Pointer
    //=====================================================================================================================

    //=====================================================================================================================
    void copyConstructVector(void* dst, const void* src)
    {
        std::construct_at(
            static_cast<MLSSvector*>(dst),
            copyVector(*static_cast<const MLSSvector*>(src))
        );
    }

    //=====================================================================================================================
    void moveConstructVector(void* dst, void* src)
    {
        std::construct_at(
            static_cast<MLSSvector*>(dst),
            moveVector(std::move(*static_cast<MLSSvector*>(src)))
        );
    }

    //=====================================================================================================================
    void destroyVector(void* ptr)
    {
        MLSSvector* vec = static_cast<MLSSvector*>(ptr);
        destroyVectorObject(*vec);

        std::destroy_at(vec);
    }

    //=====================================================================================================================
    // MLSSbinary Function Pointer
    //=====================================================================================================================

    //=====================================================================================================================
    void copyConstructBinary(void* dst, const void* src)
    {
        std::construct_at(
            static_cast<MLSSbinary*>(dst),
            copyBinary(*static_cast<const MLSSbinary*>(src))
        );
    }

    //=====================================================================================================================
    void moveConstructBinary(void* dst, void* src)
    {
        std::construct_at(
            static_cast<MLSSbinary*>(dst),
            moveBinary(std::move(*static_cast<MLSSbinary*>(src)))
        );
    }

    //=====================================================================================================================
    void destroyBinary(void* ptr)
    {
        MLSSbinary* bin = static_cast<MLSSbinary*>(ptr);
        destroyBinaryObject(*bin);

        std::destroy_at(bin);
    }

    //=====================================================================================================================
    // Any Creation Functions
    //=====================================================================================================================

    //=====================================================================================================================
    Any createVectorAny(const MLSSvector& vec)
    {
        return Any(vec, copyConstructVector, moveConstructVector, destroyVector);
    }

    //=====================================================================================================================
    Any createVectorAny(MLSSvector&& vec)
    {
        return Any(std::move(vec), copyConstructVector, moveConstructVector, destroyVector);
    }

    //=====================================================================================================================
    Any createBinaryAny(const MLSSbinary& bin)
    {
        return Any(bin, copyConstructBinary, moveConstructBinary, destroyBinary);
    }

    //=====================================================================================================================
    Any createBinaryAny(MLSSbinary&& bin)
    {
        return Any(std::move(bin), copyConstructBinary, moveConstructBinary, destroyBinary);
    }

    //=====================================================================================================================
    // MLSSvector
    //=====================================================================================================================

    //=====================================================================================================================
    MLSSvector copyVector(const MLSSvector& src)
    {
        MLSSvector dst;
        dst.m_size = src.m_size;
        dst.m_type = src.m_type;
        dst.m_handle = 0;

        if (src.m_handle != 0 && src.m_size > 0)
        {
            // Get the source Any object
            Any* src_any = MemoryManager::template getPointer<Any>(src.m_handle);
            if (src_any && src_any->hasValue())
            {
                // Copy the Any object
                Any dst_any = *src_any;

                // Store the copy in MemoryManager
                dst.m_handle = MemoryManager::addObject(std::move(dst_any));

                // Mark as initialized
                Any* new_any = MemoryManager::template getPointer<Any>(dst.m_handle);
                if (new_any)
                {
                    MemoryManager::markAsInitialized(&new_any);
                }
            }
        }

        return dst;
    }

    //=====================================================================================================================
    MLSSvector moveVector(MLSSvector&& src) noexcept
    {
        MLSSvector dst;
        dst.m_size = src.m_size;
        dst.m_type = src.m_type;
        dst.m_handle = src.m_handle;

        // Reset source
        src.m_size = 0;
        src.m_type = MLSS_NONE_TYPE;
        src.m_handle = 0;

        return dst;
    }

    //=====================================================================================================================
    void destroyVectorObject(MLSSvector& vec)
    {
        if (vec.m_handle != 0)
        {
            // The MemoryManager will handle cleanup of the Any
            vec.m_handle = 0;
        }
        vec.m_size = 0;
        vec.m_type = MLSS_NONE_TYPE;
    }

    //=====================================================================================================================
    // MLSSbinary Implementation
    //=====================================================================================================================

    //=====================================================================================================================
    MLSSbinary copyBinary(const MLSSbinary& src)
    {
        MLSSbinary dst;

        // Copy string pointers (static strings)
        dst.m_pOperatorName = src.m_pOperatorName;
        dst.m_ASIC = src.m_ASIC;
        dst.m_pKernelName = src.m_pKernelName;

        // Copy simple fields
        dst.m_grid = src.m_grid;
        dst.m_blocks = src.m_blocks;
        dst.m_sharedMemInBytes = src.m_sharedMemInBytes;

        // Deep copy vector fields
        dst.m_constants = copyVector(src.m_constants);
        dst.m_argList = copyVector(src.m_argList);

        // Copy binary data pointer
        dst.m_binaries = src.m_binaries;

        return dst;
    }

    //=====================================================================================================================
    MLSSbinary moveBinary(MLSSbinary&& src) noexcept
    {
        MLSSbinary dst;

        // Move string pointers
        dst.m_pOperatorName = src.m_pOperatorName;
        dst.m_ASIC = src.m_ASIC;
        dst.m_pKernelName = src.m_pKernelName;

        // Move simple fields
        dst.m_grid = src.m_grid;
        dst.m_blocks = src.m_blocks;
        dst.m_sharedMemInBytes = src.m_sharedMemInBytes;

        // Move vector fields
        dst.m_constants = moveVector(std::move(src.m_constants));
        dst.m_argList = moveVector(std::move(src.m_argList));

        // Move binary data
        dst.m_binaries = src.m_binaries;

        // Reset source
        src.m_pOperatorName = nullptr;
        src.m_ASIC = nullptr;
        src.m_pKernelName = nullptr;
        src.m_grid = { 0, 0, 0 };
        src.m_blocks = { 0, 0, 0 };
        src.m_sharedMemInBytes = 0;
        src.m_binaries = nullptr;

        return dst;
    }

    //=====================================================================================================================
    void destroyBinaryObject(MLSSbinary& bin)
    {
        // Don't free static strings, just nullify
        bin.m_pOperatorName = nullptr;
        bin.m_ASIC = nullptr;
        bin.m_pKernelName = nullptr;

        // Destroy vector fields
        destroyVectorObject(bin.m_constants);
        destroyVectorObject(bin.m_argList);

        // Reset other fields
        bin.m_grid = { 0, 0, 0 };
        bin.m_blocks = { 0, 0, 0 };
        bin.m_sharedMemInBytes = 0;
        bin.m_binaries = nullptr;
    }

    //=====================================================================================================================
    // Template helper for creating typed vectors in Any
    //=====================================================================================================================

    //=====================================================================================================================
    template<typename T>
    MLSSvector createTypedVectorInAny(const T* data, size_t count)
    {
        MLSSvector vec;
        vec.m_size = count;
        vec.m_type = getMLSSTypeEnum<T>();
        vec.m_handle = 0;

        if (data && count > 0)
        {
            // Create std::vector<T>
            std::vector<T> storage(data, data + count);

            // Create Any with proper function pointers
            Any any_obj(std::move(storage));

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
    template<>
    MLSSvector createTypedVector<bool>(const bool* data, size_t count)
    {
        MLSSvector vec;
        vec.m_size = count;
        vec.m_type = MLSS_BOOL;
        vec.m_handle = 0;

        if (data && count > 0)
        {
            // Convert bool to uint8_t to avoid std::vector<bool> issues
            std::vector<uint8_t> storage;
            storage.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                storage.push_back(data[i] ? 1 : 0);
            }

            Any any_obj = std::move(storage);
            vec.m_handle = MemoryManager::addObject(std::move(any_obj));

            Any* new_any = MemoryManager::template getPointer<Any>(vec.m_handle);
            if (new_any)
            {
                MemoryManager::markAsInitialized(&new_any);
            }
        }

        return vec;
    }

    //=====================================================================================================================
    bool isVectorValid(const MLSSvector& vec)
    {
        return vec.m_handle != 0 && vec.m_size > 0;
    }

} // mlss
