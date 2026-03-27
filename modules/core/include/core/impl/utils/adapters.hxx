/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    // MLSSvector Function Pointers for Any
    //=====================================================================================================================

    /// @brief Copy constructor function for MLSSvector
    void copyConstructVector(void* dst, const void* src);

    /// @brief Move constructor function for MLSSvector
    void moveConstructVector(void* dst, void* src);

    /// @brief Destructor function for MLSSvector
    void destroyVector(void* ptr);

    //=====================================================================================================================
    // MLSSbinary Function Pointers for Any
    //=====================================================================================================================

    /// @brief Copy constructor function for MLSSbinary
    void copyConstructBinary(void* dst, const void* src);

    /// @brief Move constructor function for MLSSbinary
    void moveConstructBinary(void* dst, void* src);

    /// @brief Destructor function for MLSSbinary
    void destroyBinary(void* ptr);

    //=====================================================================================================================
    // Helper functions for creating Any objects with proper function pointers
    //=====================================================================================================================

    /// @brief Create Any from MLSSvector
    Any createVectorAny(const MLSSvector& vec);
    Any createVectorAny(MLSSvector&& vec);

    /// @brief Create Any from MLSSbinary
    Any createBinaryAny(const MLSSbinary& bin);
    Any createBinaryAny(MLSSbinary&& bin);

    //=====================================================================================================================
    // MLSSvector Functions
    //=====================================================================================================================

    MLSSvector copyVector(const MLSSvector& src);
    MLSSvector moveVector(MLSSvector&& src) noexcept;
    void destroyVectorObject(MLSSvector& vec);

    //=====================================================================================================================
    // MLSSbinary Functions
    //=====================================================================================================================

    MLSSbinary copyBinary(const MLSSbinary& src);
    MLSSbinary moveBinary(MLSSbinary&& src) noexcept;
    void destroyBinaryObject(MLSSbinary& bin);

    //=====================================================================================================================
    bool isVectorValid(const MLSSvector& vec);

    //=====================================================================================================================
    // Template function declarations
    //=====================================================================================================================

    template <typename T>
    MLSSvector createTypedVector(const T* data, size_t count);

    //=====================================================================================================================
    // Template function specializations
    //=====================================================================================================================

    template <>
    MLSSvector createTypedVector<bool>(const bool* data, size_t count);

    //=====================================================================================================================
    // Template helper functions
    //=====================================================================================================================

    template <typename T>
    const T* getVectorData(const MLSSvector& vec);

    template <typename T>
    T* getVectorDataMutable(const MLSSvector& vec);

} // namespace mlss
