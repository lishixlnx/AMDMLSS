#pragma once

namespace mlss
{

    //=================================================================================================================
    //                                   Any
    //=================================================================================================================
    class Any
    {

    public:

        struct TypeInfo
        {
            std::size_t size;
            std::size_t alignment;
            bool isRange;
            void (*copyConstruct)(void* dst, const void* src);
            void (*moveConstruct)(void* dst, void* src);
            void (*destroy)(void* ptr);
        };

        //---------------------------------------------------------------------
        constexpr Any() noexcept = default;

        //---------------------------------------------------------------------
        Any(const Any& other);

        //---------------------------------------------------------------------
        Any(Any&& other) noexcept;

        //---------------------------------------------------------------------
        template <typename T>
            requires(!std::same_as<std::remove_cvref_t<T>, Any>)
        Any(T&& value);

        //---------------------------------------------------------------------
        template <typename T>
        Any(T&& value,
            void (*copyConstruct)(void*, const void*),
            void (*moveConstruct)(void*, void*),
            void (*destroy)(void*));

        //---------------------------------------------------------------------
        ~Any();

        //---------------------------------------------------------------------
        Any& operator=(const Any& other);

        //---------------------------------------------------------------------
        Any& operator=(Any&& other) noexcept;

        //---------------------------------------------------------------------
        template <typename T>
            requires(!std::same_as<std::remove_cvref_t<T>, Any>)
        Any& operator=(T&& value);

        //---------------------------------------------------------------------
        template <typename T, typename... Args>
        T& emplace(Args&&... args);

        //---------------------------------------------------------------------
        void swap(Any& other) noexcept;

        //---------------------------------------------------------------------
        [[nodiscard]] bool hasValue() const noexcept;

        //---------------------------------------------------------------------
        [[nodiscard]] bool isRange() const noexcept;

        //---------------------------------------------------------------------
        [[nodiscard]] size_t size() const noexcept;

        //---------------------------------------------------------------------
        [[nodiscard]] size_t storageSize() const noexcept;

        //---------------------------------------------------------------------
        [[nodiscard]] bool isCacheAligned() const noexcept;

        //---------------------------------------------------------------------
        template <typename T>
        friend bool anyIs(const Any& any) noexcept;

        //---------------------------------------------------------------------
        template <typename T>
        friend T* anyCast(Any* any) noexcept;

        //---------------------------------------------------------------------
        template <typename T>
        friend const T* anyCast(const Any* any) noexcept;

        //---------------------------------------------------------------------
        template <typename T>
        friend T anyCast(Any& any);

        //---------------------------------------------------------------------
        template <typename T>
        friend T anyCast(const Any& any);

        //---------------------------------------------------------------------
        template <typename T>
        friend T anyCast(Any&& any);

        //---------------------------------------------------------------------
        template <typename T>
        friend std::expected<T, std::error_code> anyCastExpected(const Any& any) noexcept;

    private:

        using CacheAlignedStorage = std::vector<uint8_t, CacheAlignedAllocator<uint8_t>>;

        //---------------------------------------------------------------------
        template <typename T>
        static void defaultCopyConstruct(void* dst, const void* src);

        //---------------------------------------------------------------------
        template <typename T>
        static void defaultMoveConstruct(void* dst, void* src);

        //---------------------------------------------------------------------
        template <typename T>
        static void defaultDestroy(void* ptr);

        //---------------------------------------------------------------------
        template <typename T>
        static constexpr bool isRangeV();

        //---------------------------------------------------------------------
        template <typename T>
        static const TypeInfo& getTypeInfo();

        //---------------------------------------------------------------------
        template <typename T>
        static std::unique_ptr<TypeInfo> createCustomTypeInfo(
            void (*copyConstruct)(void*, const void*),
            void (*moveConstruct)(void*, void*),
            void (*destroy)(void*));

        //---------------------------------------------------------------------
        void* allocateStorage(size_t size, size_t alignment);

        //---------------------------------------------------------------------
        void reset();

        //---------------------------------------------------------------------
        void* getStoragePtr() const noexcept;

        //---------------------------------------------------------------------
        CacheAlignedStorage m_storage;
        std::unique_ptr<const TypeInfo> m_typeInfo;
        mutable std::mutex m_mutex;
    };

    //=================================================================================================================
    //                                   anyIs
    //=================================================================================================================
    template <typename T>
    [[nodiscard]] bool anyIs(const Any& any) noexcept;

    //=================================================================================================================
    template <typename T>
    [[nodiscard]] T* anyCast(Any* any) noexcept;

    //=================================================================================================================
    template <typename T>
    [[nodiscard]] const T* anyCast(const Any* any) noexcept;

    //=================================================================================================================
    template <typename T>
    [[nodiscard]] T anyCast(Any& any);

    //=================================================================================================================
    template <typename T>
    [[nodiscard]] T anyCast(const Any& any);

    //=================================================================================================================
    template <typename T>
    [[nodiscard]] T anyCast(Any&& any);

    //=================================================================================================================
    template <typename T>
    [[nodiscard]] std::expected<T, std::error_code> anyCastExpected(const Any& any) noexcept;

} // namespace mlss
