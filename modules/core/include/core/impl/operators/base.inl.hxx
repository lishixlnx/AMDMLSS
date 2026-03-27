#pragma once

namespace mlss
{

    template <typename Derived, OperatorRegistration Mode>
    std::expected<std::vector<std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>>>, std::error_code> OperatorBase<Derived, Mode>::create(const Context& op)
    {

        GfxArchitectureFlags gfxFlag;

        if (auto tmp = architechtureStringToFlag(op.m_asic); tmp.has_value())
        {
            gfxFlag = tmp.value();
        }
        else
        {
            return std::unexpected(std::make_error_code(std::errc::no_such_device));
        }

        std::vector<std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>>> ops;

        ops.reserve(op.m_ops.size());

        for (auto& op : op.m_ops)
        {
            ops.emplace_back(createOp(op, gfxFlag));
        }

        return ops;
    }

    template <typename Derived, OperatorRegistration Mode>
    std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>> OperatorBase<Derived, Mode>::createOp(const Context::Op& op, GfxArchitectureFlags gfxArch)
    {
        auto instance = std::make_unique<Derived>(op.m_params, gfxArch);
        instance->m_op = std::make_unique<OperatorID>();

        return instance;
    }

    template <typename Derived, OperatorRegistration Mode>
    std::vector<std::string> OperatorBase<Derived, Mode>::getRegisteredClasses()
    {
        return std::move(OperatorRegistry::getRegisteredTypes());
    }

    template <typename Derived, OperatorRegistration Mode>
    OperatorBase<Derived, Mode>::OperatorBase(const std::vector<mlss::Attribute>& attributes, GfxArchitectureFlags gfxip)
        : m_attributes(attributes), m_gfxArch(gfxip)
    {
    }

    template <typename Derived, OperatorRegistration Mode>
    void OperatorBase<Derived, Mode>::setAttributes(const std::vector<mlss::Attribute>& attributes)
    {
        m_attributes = attributes;
    }

    template <typename Derived, OperatorRegistration Mode>
    void OperatorBase<Derived, Mode>::setGfxArchitecture(GfxArchitectureFlags gfxArch)
    {
        m_gfxArch = gfxArch;
    }

    template <typename Derived, OperatorRegistration Mode>
    std::string OperatorBase<Derived, Mode>::getImplName() const
    {
        return m_implName;
    }

    template <typename Derived, OperatorRegistration Mode>
    bool OperatorBase<Derived, Mode>::registerInstance(const std::string& name)
    {
        if constexpr (Mode == OperatorRegistration::Enabled)
        {
            OperatorRegistry::registerType(name, []() -> std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>>
                                           { return std::unique_ptr<OperatorBase<void, OperatorRegistration::Enabled>>(
                                                 reinterpret_cast<OperatorBase<void, OperatorRegistration::Enabled>*>(new Derived())); });
        }
        return true;
    }

    // Static member initialization
    template <typename Derived, OperatorRegistration Mode>
    bool OperatorBase<Derived, Mode>::registered = OperatorBase<Derived, Mode>::registerInstance(Derived::getOperatorName());

} // namespace mlss
