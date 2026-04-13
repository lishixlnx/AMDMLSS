/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <unordered_map>

namespace mlss
{

    template <typename OperatorType>
    class CaseRegistry
    {
    public:

        using GetCapsFunction = std::function<uint32_t(const std::vector<Attribute>&, GfxIpTriple, const void*)>;
        using GetBinariesFunction = std::function<std::expected<Binaries, std::error_code>(const std::vector<Attribute>&, const GfxIpTriple&)>;

        struct Entry
        {
            std::string name;
            GetCapsFunction getCaps;
            GetBinariesFunction getBinaries;
        };

        template <typename CaseType>
        static void registerCase(Entry entry)
        {
            registry().emplace_back(entry);
            typedRegistry().emplace(typeKey<CaseType>(), std::move(entry));
        }

        static const std::vector<Entry>& getCases()
        {
            return registry();
        }

        template <typename CaseType>
        static const Entry* get()
        {
            auto it = typedRegistry().find(typeKey<CaseType>());
            if (it != typedRegistry().end())
            {
                return &it->second;
            }
            return nullptr;
        }

    private:

        static std::vector<Entry>& registry()
        {
            static std::vector<Entry> instance;
            return instance;
        }

        static std::unordered_map<TypeKey, Entry, TypeKeyHash>& typedRegistry()
        {
            static std::unordered_map<TypeKey, Entry, TypeKeyHash> instance;
            return instance;
        }
    };

    template <typename Derived, typename OperatorType>
    class CaseBase
    {
    public:

        virtual ~CaseBase() = default;

        virtual std::expected<Binaries, std::error_code> getBinaries() const = 0;

        static uint32_t getCaps(const std::vector<Attribute>& attributes, GfxIpTriple gfxIp, const void* context = nullptr)
        {
            if constexpr (requires { { Derived::getCapsImpl(attributes, gfxIp, context) } -> std::convertible_to<uint32_t>; })
            {
                return static_cast<uint32_t>(Derived::getCapsImpl(attributes, gfxIp, context));
            }
            else if constexpr (requires { { Derived::getCapsImpl(attributes, gfxIp) } -> std::convertible_to<uint32_t>; })
            {
                return static_cast<uint32_t>(Derived::getCapsImpl(attributes, gfxIp));
            }
            else if constexpr (requires { { Derived::getCapsImpl(attributes) } -> std::convertible_to<uint32_t>; })
            {
                std::ignore = gfxIp;
                return static_cast<uint32_t>(Derived::getCapsImpl(attributes));
            }
            else
            {
                static_assert(sizeof(Derived) == 0,
                              "Derived class must implement getCapsImpl with signature "
                              "'static uint32_t getCapsImpl(const std::vector<Attribute>&)' or "
                              "'static uint32_t getCapsImpl(const std::vector<Attribute>&, GfxIpTriple)' or "
                              "'static uint32_t getCapsImpl(const std::vector<Attribute>&, GfxIpTriple, const void*)'");
                return 0x00000000u;
            }
        }

        static bool registerCase()
        {
            CaseRegistry<OperatorType>::template registerCase<Derived>(
                {Derived::getCaseName(),
                 [](const std::vector<Attribute>& attrs, GfxIpTriple arch, const void* context) -> uint32_t
                 {
                     return Derived::getCaps(attrs, arch, context);
                 },
                 [](const std::vector<Attribute>& attrs, const GfxIpTriple& arch) -> std::expected<Binaries, std::error_code>
                 {
                     Derived caseInstance(attrs, arch);
                     return caseInstance.getBinaries();
                 }});
            return true;
        }

        void setAttributes(const std::vector<Attribute>& attributes)
        {
            m_attributes = attributes;
        }

        void setGfxIpTriple(GfxIpTriple gfxIp)
        {
            m_gfxIpTriple = gfxIp;
        }

        std::string getImplName() const
        {
            return m_implName;
        }

    protected:

        constexpr inline CaseBase() : m_gfxIpTriple(IP_GFX_UNKNOWN) { (void)s_registered; }

        CaseBase(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
            : m_attributes(attributes), m_gfxIpTriple(gfxip)
        {
            (void)s_registered;
        }

        std::vector<Attribute> m_attributes;
        GfxIpTriple m_gfxIpTriple;
        mutable std::string m_implName;

    private:

        inline static bool s_registered = registerCase();
    };

    template <typename OperatorType>
    struct CaseSelector
    {
        struct SelectionResult
        {
            std::expected<Binaries, std::error_code> binaries;
            std::string implName;
        };

        static bool anyCaps(const std::vector<Attribute>& attrs, GfxIpTriple arch, const void* context = nullptr)
        {
            for (const auto& entry : CaseRegistry<OperatorType>::getCases())
            {
                if (entry.getCaps(attrs, arch, context) != 0x00000000u)
                {
                    return true;
                }
            }
            return false;
        }

        static uint32_t bestCaps(const std::vector<Attribute>& attrs, GfxIpTriple arch, const void* context = nullptr)
        {
            uint32_t best = 0x00000000u;
            for (const auto& entry : CaseRegistry<OperatorType>::getCases())
            {
                uint32_t score = entry.getCaps(attrs, arch, context);
                if (score > best)
                {
                    best = score;
                }
            }
            return best;
        }

        static SelectionResult select(const std::vector<Attribute>& attrs, const GfxIpTriple& arch, const void* context = nullptr)
        {
            uint32_t bestScore = 0x00000000u;
            const typename CaseRegistry<OperatorType>::Entry* bestEntry = nullptr;

            for (const auto& entry : CaseRegistry<OperatorType>::getCases())
            {
                uint32_t score = entry.getCaps(attrs, arch, context);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestEntry = &entry;
                }
            }

            if (bestEntry != nullptr)
            {
                return {bestEntry->getBinaries(attrs, arch), bestEntry->name};
            }

            return {std::unexpected(std::make_error_code(std::errc::not_supported)), {}};
        }
    };

} // namespace mlss
