#pragma once

namespace mlss
{

    //=====================================================================================================================
    template <class T>
    T& Attribute::value()
    {
        if (getTypeFlag<T>() != m_info->type_enum())
        {
            throw std::runtime_error(std::format("{}: Type mismatch!", __func__));
        }
        return *reinterpret_cast<T*>(m_attr.data());
    }
    //=====================================================================================================================
    template <class T>
    const T& Attribute::value() const
    {

        if (getTypeFlag<T>() != m_info->type_enum())
        {
            throw std::runtime_error(std::format("{}: Type mismatch!", __func__));
        }

        return *reinterpret_cast<const T*>(m_attr.data());
    }

    //=====================================================================================================================
    template <class T>
    T* Attribute::data()
    {
        if (getTypeFlag<T>() != makeArrayEnum(m_info->element_type_enum()))
        {
            throw std::runtime_error(std::format("{}: Type mismatch!", __func__));
        }
        return reinterpret_cast<T*>(m_attr.data());
    }

    //=====================================================================================================================
    template <class T>
    const T* Attribute::data() const
    {
        if (getTypeFlag<T>() != makeArrayEnum(m_info->element_type_enum()))
        {
            throw std::runtime_error(std::format("{}: Type mismatch!", __func__));
        }
        return reinterpret_cast<const T*>(m_attr.data());
    }

} // namespace mlss
