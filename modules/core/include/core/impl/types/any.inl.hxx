#pragma once

namespace mlss
{
    //=====================================================================================================================
    //                                   Any
    //=====================================================================================================================

    //---------------------------------------------------------------------
    template <typename T>
    void Any::defaultCopyConstruct(void* dst, const void* src)
    {
        if constexpr (std::copy_constructible<T>)
        {
            std::construct_at(static_cast<T*>(dst), *static_cast<const T*>(src));
        }
    }

    //---------------------------------------------------------------------
    template <typename T>
    void Any::defaultMoveConstruct(void* dst, void* src)
    {
        if constexpr (std::move_constructible<T>)
        {
            std::construct_at(static_cast<T*>(dst), std::move(*static_cast<T*>(src)));
        }
    }

    //---------------------------------------------------------------------
    template <typename T>
    void Any::defaultDestroy(void* ptr)
    {
        if constexpr (!std::is_trivially_destructible_v<T>)
        {
            std::destroy_at(static_cast<T*>(ptr));
        }
    }

    //---------------------------------------------------------------------
    template <typename T>
    constexpr bool Any::isRangeV()
    {
        return requires(T& t) {
            std::begin(t);
            std::end(t);
        };
    }

    //---------------------------------------------------------------------
    template <typename T>
    const Any::TypeInfo& Any::getTypeInfo()
    {
        static const TypeInfo info =
            {
                sizeof(T),
                alignof(T),
                isRangeV<T>(),
                &defaultCopyConstruct<T>,
                &defaultMoveConstruct<T>,
                &defaultDestroy<T>};
        return info;
    }

    //---------------------------------------------------------------------
    template <typename T>
    std::unique_ptr<Any::TypeInfo> Any::createCustomTypeInfo(
        void (*copyConstruct)(void*, const void*),
        void (*moveConstruct)(void*, void*),
        void (*destroy)(void*))
    {
        return std::make_unique<TypeInfo>(TypeInfo{
            sizeof(T),
            alignof(T),
            isRangeV<T>(),
            copyConstruct ? copyConstruct : &defaultCopyConstruct<T>,
            moveConstruct ? moveConstruct : &defaultMoveConstruct<T>,
            destroy ? destroy : &defaultDestroy<T>});
    }

    //---------------------------------------------------------------------
    template <typename T>
        requires(!std::same_as<std::remove_cvref_t<T>, Any>)
    Any::Any(T&& value)
    {
        emplace<std::remove_cvref_t<T>>(std::forward<T>(value));
    }

    //---------------------------------------------------------------------
    template <typename T>
    Any::Any(T&& value,
             void (*copyConstruct)(void*, const void*),
             void (*moveConstruct)(void*, void*),
             void (*destroy)(void*))
    {
        using DecayedT = std::remove_cvref_t<T>;

        m_typeInfo = createCustomTypeInfo<DecayedT>(
            copyConstruct, moveConstruct, destroy);

        void* storagePtr = allocateStorage(m_typeInfo->size, m_typeInfo->alignment);

        if (storagePtr)
        {
            if constexpr (std::is_rvalue_reference_v<decltype(std::forward<T>(value))>)
            {
                m_typeInfo->moveConstruct(storagePtr, &value);
            }
            else
            {
                m_typeInfo->copyConstruct(storagePtr, &value);
            }
        }
    }

    //---------------------------------------------------------------------
    template <typename T>
        requires(!std::same_as<std::remove_cvref_t<T>, Any>)
    Any& Any::operator=(T&& value)
    {
        emplace<std::remove_cvref_t<T>>(std::forward<T>(value));
        return *this;
    }

    //---------------------------------------------------------------------
    template <typename T, typename... Args>
    T& Any::emplace(Args&&... args)
    {
        reset();

        using DecayedT = std::remove_cvref_t<T>;

        std::lock_guard<std::mutex> lock(m_mutex);
        // Create a copy of the static TypeInfo
        m_typeInfo = std::make_unique<TypeInfo>(getTypeInfo<DecayedT>());

        void* storagePtr = allocateStorage(m_typeInfo->size, m_typeInfo->alignment);

        if (!storagePtr)
        {
            throw std::bad_alloc();
        }

        // Construct the object in-place using std::construct_at
        T* objPtr = std::construct_at(static_cast<T*>(storagePtr), std::forward<Args>(args)...);
        return *objPtr;
    }

    //=====================================================================================================================
    //                                   anyIs
    //=====================================================================================================================
    template <typename T>
    bool anyIs(const Any& any) noexcept
    {
        using DecayedT = std::remove_cvref_t<T>;
        std::lock_guard<std::mutex> lock(any.m_mutex);

        if (!any.m_typeInfo || any.m_storage.empty())
        {
            return false;
        }

        // Compare with standard type info
        const auto& stdTypeInfo = Any::getTypeInfo<DecayedT>();
        return any.m_typeInfo->size == stdTypeInfo.size &&
               any.m_typeInfo->alignment == stdTypeInfo.alignment;
    }

    //=====================================================================================================================
    //                                   anyCast
    //=====================================================================================================================

    //=====================================================================================================================
    template <typename T>
    T* anyCast(Any* any) noexcept
    {
        using DecayedT = std::remove_cvref_t<T>;
        if (any && anyIs<DecayedT>(*any))
        {
            std::lock_guard<std::mutex> lock(any->m_mutex);
            return static_cast<DecayedT*>(any->getStoragePtr());
        }
        return nullptr;
    }

    //=====================================================================================================================
    template <typename T>
    const T* anyCast(const Any* any) noexcept
    {
        using DecayedT = std::remove_cvref_t<T>;
        if (any && anyIs<DecayedT>(*any))
        {
            std::lock_guard<std::mutex> lock(any->m_mutex);
            return static_cast<const DecayedT*>(any->getStoragePtr());
        }
        return nullptr;
    }

    //=====================================================================================================================
    template <typename T>
    T anyCast(Any& any)
    {
        using DecayedT = std::remove_cvref_t<T>;
        auto result = anyCast<DecayedT>(&any);
        if (!result)
        {
            throw std::bad_cast();
        }

        if constexpr (std::is_lvalue_reference_v<T>)
        {
            return static_cast<T>(*result);
        }
        else
        {
            return static_cast<T>(std::move(*result));
        }
    }

    //=====================================================================================================================
    template <typename T>
    T anyCast(const Any& any)
    {
        using DecayedT = std::remove_cvref_t<T>;
        auto result = anyCast<DecayedT>(&any);
        if (!result)
        {
            throw std::bad_cast();
        }
        return static_cast<T>(*result);
    }

    //=====================================================================================================================
    template <typename T>
    T anyCast(Any&& any)
    {
        using DecayedT = std::remove_cvref_t<T>;
        auto result = anyCast<DecayedT>(&any);
        if (!result)
        {
            throw std::bad_cast();
        }
        return static_cast<T>(std::move(*result));
    }

    //=====================================================================================================================
    //                                   anyCastExpected
    //=====================================================================================================================
    template <typename T>
    std::expected<T, std::error_code> anyCastExpected(const Any& any) noexcept
    {
        using DecayedT = std::remove_cvref_t<T>;
        auto result = anyCast<DecayedT>(&any);
        if (!result)
        {
            return std::unexpected(std::make_error_code(std::errc::invalid_argument));
        }
        return static_cast<T>(*result);
    }

} // namespace mlss
