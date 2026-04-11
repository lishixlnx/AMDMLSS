/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{

    template <typename OperatorType>
    class BackendRegistry
    {
    public:

        using GetCapsFunction = std::function<bool(const std::vector<Attribute>&, GfxIpTriple)>;
        using GetBinariesFunction = std::function<std::expected<Binaries, std::error_code>(const std::vector<Attribute>&, const GfxIpTriple&)>;

        struct Entry
        {
            std::string name;
            GetCapsFunction getCaps;
            GetBinariesFunction getBinaries;
        };

        static void registerBackend(Entry entry)
        {
            registry().emplace_back(std::move(entry));
        }

        static const std::vector<Entry>& getBackends()
        {
            return registry();
        }

    private:

        static std::vector<Entry>& registry()
        {
            static std::vector<Entry> instance;
            return instance;
        }
    };

    template <typename Derived, typename OperatorType>
    class BackendBase
    {
    public:

        virtual ~BackendBase() = default;

        virtual std::expected<Binaries, std::error_code> getBinaries() const = 0;

        static bool getCaps(const std::vector<Attribute>& attributes, GfxIpTriple gfxIp)
        {
            if constexpr (requires { { Derived::getCapsImpl(attributes, gfxIp) } -> std::convertible_to<bool>; })
            {
                return Derived::getCapsImpl(attributes, gfxIp);
            }
            else if constexpr (requires { { Derived::getCapsImpl(attributes) } -> std::convertible_to<bool>; })
            {
                std::ignore = gfxIp;
                return Derived::getCapsImpl(attributes);
            }
            else
            {
                static_assert(sizeof(Derived) == 0,
                              "Derived class must implement getCapsImpl with signature "
                              "'static bool getCapsImpl(const std::vector<Attribute>&)' or "
                              "'static bool getCapsImpl(const std::vector<Attribute>&, GfxIpTriple)'");
                return false;
            }
        }

        static bool registerBackend()
        {
            BackendRegistry<OperatorType>::registerBackend(
                {Derived::getOperatorName(),
                 [](const std::vector<Attribute>& attrs, GfxIpTriple arch) -> bool
                 {
                     return Derived::getCaps(attrs, arch);
                 },
                 [](const std::vector<Attribute>& attrs, const GfxIpTriple& arch) -> std::expected<Binaries, std::error_code>
                 {
                     Derived backend(attrs, arch);
                     return backend.getBinaries();
                 }});
            return true;
        }

        static bool hasDecisionFunction()
        {
            return m_decision != nullptr;
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

        constexpr inline BackendBase() : m_gfxIpTriple(IP_GFX_UNKNOWN) { (void)s_registered; }

        BackendBase(const std::vector<Attribute>& attributes, const GfxIpTriple& gfxip)
            : m_attributes(attributes), m_gfxIpTriple(gfxip)
        {
            (void)s_registered;
        }

        std::vector<Attribute> m_attributes;
        GfxIpTriple m_gfxIpTriple;
        mutable std::string m_implName;

        inline static std::function<void(const std::vector<Attribute>&, const GfxIpTriple&)> m_decision = nullptr;

    private:

        inline static bool s_registered = registerBackend();
    };

    template <typename OperatorType>
    struct BackendSelector
    {
        struct SelectionResult
        {
            std::expected<Binaries, std::error_code> binaries;
            std::string implName;
        };

        static bool anyCaps(const std::vector<Attribute>& attrs, GfxIpTriple arch)
        {
            for (const auto& backend : BackendRegistry<OperatorType>::getBackends())
            {
                if (backend.getCaps(attrs, arch))
                {
                    return true;
                }
            }
            return false;
        }

        static SelectionResult select(const std::vector<Attribute>& attrs, const GfxIpTriple& arch)
        {
            for (const auto& backend : BackendRegistry<OperatorType>::getBackends())
            {
                if (backend.getCaps(attrs, arch))
                {
                    return {backend.getBinaries(attrs, arch), backend.name};
                }
            }
            return {std::unexpected(std::make_error_code(std::errc::not_supported)), {}};
        }
    };

} // namespace mlss
