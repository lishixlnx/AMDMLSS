/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{

    template <typename Derived>
    std::expected<std::vector<std::unique_ptr<OperatorBase<void>>>, std::error_code> OperatorBase<Derived>::create(const Context& op)
    {

        GfxIpTriple gfxIp;

        if (auto tmp = architectureStringToGfxIpTriple(op.m_asic); tmp.has_value())
        {
            gfxIp = tmp.value();
        }
        else
        {
            return std::unexpected(std::make_error_code(std::errc::no_such_device));
        }

        std::vector<std::unique_ptr<OperatorBase<void>>> ops;

        ops.reserve(op.m_ops.size());

        for (auto& op : op.m_ops)
        {
            ops.emplace_back(createOp(op, gfxIp));
        }

        return ops;
    }

    template <typename Derived>
    std::unique_ptr<OperatorBase<void>> OperatorBase<Derived>::createOp(const Context::Op& op, GfxIpTriple gfxIp)
    {
        auto instance = std::make_unique<Derived>(op.m_params, gfxIp);
        instance->m_op = std::make_unique<OperatorID>();

        return instance;
    }

    template <typename Derived>
    std::vector<std::string> OperatorBase<Derived>::getRegisteredClasses()
    {
        return std::move(OperatorRegistry::getRegisteredTypes());
    }

    template <typename Derived>
    OperatorBase<Derived>::OperatorBase(const std::vector<mlss::Attribute>& attributes, GfxIpTriple gfxip)
        : m_attributes(attributes), m_gfxIpTriple(gfxip)
    {
    }

    template <typename Derived>
    void OperatorBase<Derived>::setAttributes(const std::vector<mlss::Attribute>& attributes)
    {
        m_attributes = attributes;
    }

    template <typename Derived>
    void OperatorBase<Derived>::setGfxIpTriple(GfxIpTriple gfxIp)
    {
        m_gfxIpTriple = gfxIp;
    }

    template <typename Derived>
    std::string OperatorBase<Derived>::getImplName() const
    {
        return m_implName;
    }

    template <typename Derived>
    bool OperatorBase<Derived>::registerInstance(const std::string& name)
    {
        OperatorRegistry::registerType(name, []() -> std::unique_ptr<OperatorBase<void>>
                                       { return std::unique_ptr<OperatorBase<void>>(
                                             reinterpret_cast<OperatorBase<void>*>(new Derived())); });
        return true;
    }

    template <typename Derived>
    bool OperatorBase<Derived>::registered = OperatorBase<Derived>::registerInstance(Derived::getOperatorName());

} // namespace mlss
