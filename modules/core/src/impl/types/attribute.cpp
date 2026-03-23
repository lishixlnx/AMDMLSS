#include "core/core.hpp"

namespace mlss
{

    //=====================================================================================================================
    Attribute& Attribute::operator=(const void* value)
    {
        std::copy_n(static_cast<const uint8_t*>(value), this->m_attr.size(), this->m_attr.begin());

        return (*this);
    }

    //=====================================================================================================================
    std::string Attribute::name() const
    {
        return this->m_info->name();
    }


    //=====================================================================================================================
    std::string Attribute::range() const
    {
        return this->m_info->range();
    }

    //=====================================================================================================================
    std::uint32_t Attribute::attr_enum() const
    {
        return m_info->m_attr_enum;
    }

    //=====================================================================================================================
    enum64 Attribute::type_enum() const
    {
        return m_info->m_type_enum;
    }


    //=====================================================================================================================
    bool Attribute::isArray() const
    {
        return m_info->isArray();
    }

    //=====================================================================================================================
    std::uint32_t Attribute::element_type_enum() const
    {
        return m_info->element_type_enum();
    }

    //=====================================================================================================================
    size_t Attribute::num_elements() const
    {
        return m_info->num_elements();
    }

    //=====================================================================================================================
    uint8_t* Attribute::data()
    {
        return m_attr.data();
    }

    //=====================================================================================================================
    const uint8_t* Attribute::data() const
    {
        return m_attr.data();
    }

    //=====================================================================================================================
    bool Attribute::is(const std::string& name) const
    {
        return (m_info->m_name == name);
    }

    //=====================================================================================================================
    bool Attribute::is(const std::uint32_t& flag) const
    {
        return m_info->m_attr_enum == flag;
    }

    //=====================================================================================================================
    bool operator==(const Attribute& lhs, const std::string& rhs)
    {
        return lhs.is(rhs);
    }

    //=====================================================================================================================
    bool operator ==(const std::string& lhs, const Attribute& rhs)
    {
        return rhs.is(lhs);
    }

    //=====================================================================================================================
    bool operator!=(const Attribute& lhs, const std::string& rhs)
    {
        return !lhs.is(rhs);
    }

    //=====================================================================================================================
    bool operator!=(const std::string& lhs, const Attribute& rhs)
    {
        return !rhs.is(lhs);
    }

    //=====================================================================================================================
    bool operator==(const Attribute& lhs, const uint32_t& rhs)
    {
        return lhs.is(rhs);
    }

    //=====================================================================================================================
    bool operator ==(const uint32_t& lhs, const Attribute& rhs)
    {
        return rhs.is(lhs);
    }

    //=====================================================================================================================
    bool operator!=(const Attribute& lhs, const uint32_t& rhs)
    {
        return !lhs.is(rhs);
    }

    //=====================================================================================================================
    bool operator!=(const uint32_t& lhs, const Attribute& rhs)
    {
        return !rhs.is(lhs);
    }



    //=================================================================================================================
    //                                   AttributeInfo                                                                   
    //=================================================================================================================

    //=====================================================================================================================
    Attribute::AttributeInfo::AttributeInfo(
        const std::string& name,
        const std::string& range,
        const std::uint32_t& attr_enum,
        const enum64& type_enum) :
        m_name(name),
        m_range(range),
        m_attr_enum(attr_enum),
        m_type_enum(type_enum)
    {
    }

    //=====================================================================================================================
    std::string& Attribute::AttributeInfo::name()
    {
        return this->m_name;
    }

    //=====================================================================================================================
    const std::string& Attribute::AttributeInfo::name() const
    {
        return this->m_name;
    }

    //=====================================================================================================================
    std::string& Attribute::AttributeInfo::range()
    {
        return this->m_range;
    }

    //=====================================================================================================================
    const std::string& Attribute::AttributeInfo::range() const
    {
        return this->m_range;
    }

    //=====================================================================================================================
    std::uint32_t& Attribute::AttributeInfo::attr_enum()
    {
        return this->m_attr_enum;
    }

    //=====================================================================================================================
    const std::uint32_t& Attribute::AttributeInfo::attr_enum() const
    {
        return this->m_attr_enum;
    }

    //=====================================================================================================================
    enum64& Attribute::AttributeInfo::type_enum()
    {
        return this->m_type_enum;
    }

    //=====================================================================================================================
    const enum64& Attribute::AttributeInfo::type_enum() const
    {
        return this->m_type_enum;
    }

    //=====================================================================================================================
    bool Attribute::AttributeInfo::isArray() const
    {
        return (getArrayEnumType(this->m_type_enum) & MLSS_ARRAY) == MLSS_ARRAY;
    }

    //=====================================================================================================================
    std::uint32_t Attribute::AttributeInfo::element_type_enum() const
    {
        return getArrayEnumElementType(this->m_type_enum);
    }

    //=====================================================================================================================
    size_t Attribute::AttributeInfo::num_elements() const
    {
        return getArrayEnumSize(this->m_type_enum);
    }

} // mlss
