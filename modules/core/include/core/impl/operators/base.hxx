#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>
#include <concepts>

#include "../core.hxx"

namespace mlss
{

    enum class OperatorRegistration
    {
        Disabled,
        Enabled
    };

    // Forward declare template class
    template<class Derived, OperatorRegistration Mode = OperatorRegistration::Enabled>
    class OperatorBase;

    using FactoryFunction = std::function<std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>>()>;

    class OperatorRegistry
    {
    public:

        virtual ~OperatorRegistry() = default;

        static void registerType(const std::string& name, FactoryFunction factory);

        static std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>> create(const std::string& typeName);

        static std::vector<std::string> getRegisteredTypes();

        static bool isTypeRegistered(const std::string& typeName);

    private:

        static std::unordered_map<std::string, FactoryFunction> registry;

    };
    

    // Concept to check if Derived has getCapsImpl with gfxArch parameter
    template<typename T>
    concept HasGetCapsImplWithGfxArch = requires(const std::vector<Attribute>& attrs, GfxArchitectureFlags arch) {
        { T::getCapsImpl(attrs, arch) } -> std::convertible_to<bool>;
    };

    // Concept to check if Derived has getCapsImpl without gfxArch parameter
    template<typename T>
    concept HasGetCapsImplWithoutGfxArch = requires(const std::vector<Attribute>& attrs) {
        { T::getCapsImpl(attrs) } -> std::convertible_to<bool>;
    };

    template<class Derived, OperatorRegistration Mode>
    class OperatorBase
    {
    public:

        using blob = typename Binaries::Blob;

        struct OperatorID
        {
            std::string name;
            uint32_t version;
            // Add other identification fields as needed
        };

        static bool getCaps(const std::vector<Attribute>& attributes, GfxArchitectureFlags gfxArch)
        {
            if constexpr (HasGetCapsImplWithGfxArch<Derived>)
            {
                return Derived::getCapsImpl(attributes, gfxArch);
            }
             else if constexpr (HasGetCapsImplWithoutGfxArch<Derived>)
             {
                std::ignore = gfxArch;
                return Derived::getCapsImpl(attributes);
            }
            else
            {
                static_assert(HasGetCapsImplWithGfxArch<Derived> || HasGetCapsImplWithoutGfxArch<Derived>,
                             "Derived class must implement getCapsImpl with either (attributes, gfxArch) or (attributes) signature");
                return false;
            }
        }

        // Static method to check capabilities
        static bool getCaps(const std::vector<Attribute>& attributes)
        {
            return Derived::getCapsImpl(attributes);
        }

        // Factory method to create instances
        static std::expected<std::vector<std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>>>, std::error_code> create(const Context& op);

        static std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>> createOp(const Context::Op& op, GfxArchitectureFlags gfxArch);

        virtual ~OperatorBase() = default;

        // Pure virtual method to get the binary blob
        virtual std::expected<blob, std::error_code> getBlob() const = 0;

        // Register this operator type with the registry
        static bool registerInstance(const std::string& name);

        // Get all registered derived class names
        static std::vector<std::string> getRegisteredClasses();

        // Set attributes for this operator
        void setAttributes(const std::vector<mlss::Attribute>& attributes);

        // Set graphics architecture
        void setGfxArchitecture(GfxArchitectureFlags gfxArch);

        std::string getImplName() const;

    protected:

        // Protected constructors
        constexpr inline OperatorBase() : m_gfxArch(GfxArchitectureFlags::Unknown) {}
        
        OperatorBase(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip);

        std::vector<mlss::Attribute> m_attributes;
        GfxArchitectureFlags m_gfxArch;
        std::unique_ptr<OperatorID> m_op;
        mutable std::string m_implName;
        

    private:

        static bool registered;

    };

} // mlss

#include "base.inl.hxx"
