/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "../core.hxx"

namespace mlss
{

    template <class Derived>
    class OperatorBase;

    using FactoryFunction = std::function<std::unique_ptr<OperatorBase<void>>()>;

    class OperatorRegistry
    {
    public:

        virtual ~OperatorRegistry() = default;

        static void registerType(const std::string& name, FactoryFunction factory);

        static std::unique_ptr<OperatorBase<void>> create(const std::string& typeName);

        static std::vector<std::string> getRegisteredTypes();

        static bool isTypeRegistered(const std::string& typeName);

    private:

        static std::unordered_map<std::string, FactoryFunction> registry;
    };

    template <typename T>
    concept HasGetCapsImplWithGfxIpTriple = requires(const std::vector<Attribute>& attrs, GfxIpTriple arch) {
        { T::getCapsImpl(attrs, arch) } -> std::convertible_to<bool>;
    };

    template <typename T>
    concept HasGetCapsImplWithoutGfxArch = requires(const std::vector<Attribute>& attrs) {
        { T::getCapsImpl(attrs) } -> std::convertible_to<bool>;
    };

    template <class Derived>
    class OperatorBase
    {
    public:

        using blob = typename Binaries::Blob;

        struct OperatorID
        {
            std::string name;
            uint32_t version;
        };

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

        static bool getCaps(const std::vector<Attribute>& attributes)
        {
            return Derived::getCapsImpl(attributes);
        }

        static std::expected<std::vector<std::unique_ptr<OperatorBase<void>>>, std::error_code> create(const Context& op);

        static std::unique_ptr<OperatorBase<void>> createOp(const Context::Op& op, GfxIpTriple gfxIp);

        virtual ~OperatorBase() = default;

        virtual std::expected<Binaries, std::error_code> getBinaries() const = 0;

        static bool registerInstance(const std::string& name);

        static std::vector<std::string> getRegisteredClasses();

        void setAttributes(const std::vector<mlss::Attribute>& attributes);

        void setGfxIpTriple(GfxIpTriple gfxIp);

        std::string getImplName() const;

    protected:

        constexpr inline OperatorBase() : m_gfxIpTriple(IP_GFX_UNKNOWN) {}

        OperatorBase(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip);

        std::vector<mlss::Attribute> m_attributes;
        GfxIpTriple m_gfxIpTriple;
        std::unique_ptr<OperatorID> m_op;
        mutable std::string m_implName;

    private:

        static bool registered;
    };

} // namespace mlss

#include "base.inl.hxx"
#include "backend.hxx"
