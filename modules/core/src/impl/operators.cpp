/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "core/core.hpp"

namespace mlss
{
    //=================================================================================================================
    // OperatorRegistry implementation
    //=================================================================================================================

    // Static member definition
    std::unordered_map<std::string, FactoryFunction> OperatorRegistry::registry;

    //=================================================================================================================
    void OperatorRegistry::registerType(const std::string& name, FactoryFunction factory)
    {
        registry[name] = factory;
    }

    //=================================================================================================================
    std::unique_ptr<OperatorBase<void>> OperatorRegistry::create(const std::string& typeName)
    {
        auto it = registry.find(typeName);
        if (it != registry.end())
        {
            return it->second();
        }
        return nullptr;
    }

    //=================================================================================================================
    std::vector<std::string> OperatorRegistry::getRegisteredTypes()
    {
        std::vector<std::string> types;
        types.reserve(registry.size());
        for (const auto& [name, factory] : registry)
        {
            types.push_back(name);
        }
        return types;
    }

    //=================================================================================================================
    bool OperatorRegistry::isTypeRegistered(const std::string& typeName)
    {
        return registry.find(typeName) != registry.end();
    }

} // namespace mlss
