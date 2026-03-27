#include "core/core.hpp"

namespace mlss
{

    //=================================================================================================================
    //                                   enum64
    //=================================================================================================================

    //---------------------------------------------------------------------
    enum64::enum64(const uint32_t& low, const uint32_t& high) : m_attr{(static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32))}
    {
    }

    //---------------------------------------------------------------------
    enum64::enum64(const uint64_t& obj) : m_attr{obj}
    {
    }

    //---------------------------------------------------------------------
    uint32_t enum64::low() const
    {
        return m_attr.parts.low;
    }

    //---------------------------------------------------------------------
    uint32_t enum64::high() const
    {
        return m_attr.parts.high;
    }

    //---------------------------------------------------------------------
    void enum64::setLow(const uint32_t& low)
    {
        m_attr.parts.low = low;
    }

    //---------------------------------------------------------------------
    void enum64::setHigh(const uint32_t& high)
    {
        m_attr.parts.high = high;
    }

    //---------------------------------------------------------------------
    bool enum64::operator==(const enum64& other) const
    {
        // Fast path: exact match
        if (m_attr.u64 == other.m_attr.u64)
        {
            return true;
        }

        // Special comparison logic for array types with UINT32/ENUM compatibility
        auto lhs_size = getArrayEnumSize(*this);
        auto rhs_size = getArrayEnumSize(other);

        if (lhs_size != rhs_size)
        {
            return false;
        }

        auto lhs_element_type = getArrayEnumType(*this) & ~MLSS_ARRAY;
        auto rhs_element_type = getArrayEnumType(other) & ~MLSS_ARRAY;

        // Check if both are UINT32 or ENUM (they are compatible)
        bool lhs_is_uint_or_enum = (lhs_element_type == MLSS_UINT32) || (lhs_element_type == MLSS_ENUM);
        bool rhs_is_uint_or_enum = (rhs_element_type == MLSS_UINT32) || (rhs_element_type == MLSS_ENUM);

        return lhs_is_uint_or_enum && rhs_is_uint_or_enum;
    }

    //---------------------------------------------------------------------
    bool enum64::operator!=(const enum64& other) const
    {
        return !(*this == other);
    }

    //---------------------------------------------------------------------
    std::strong_ordering enum64::operator<=>(const enum64& other) const
    {
        // Compare the 64-bit values directly
        if (m_attr.u64 < other.m_attr.u64)
        {
            return std::strong_ordering::less;
        }
        else if (m_attr.u64 > other.m_attr.u64)
        {
            return std::strong_ordering::greater;
        }
        else
        {
            return std::strong_ordering::equal;
        }
    }

} // namespace mlss
