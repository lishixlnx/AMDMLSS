#pragma once

#include <cstdint>
#include <compare>

namespace mlss
{    
    //=================================================================================================================
    //                                   Enum64_t                                                                   
    //=================================================================================================================
    struct enum64
    {
        union
        {
            uint64_t u64;
            struct
            {
                uint32_t low, high; // size in the least significant 32 bits, type in the most significant 32 bits.
            } parts;
        } m_attr;

        enum64() = default;
        enum64(const uint32_t& low, const uint32_t& high);
        enum64(const uint64_t& obj);

        uint32_t low() const;
        uint32_t high() const;

        void setLow(const uint32_t& low);
        void setHigh(const uint32_t& high);

        // Comparison operators
        bool operator==(const enum64& other) const;
        bool operator!=(const enum64& other) const;
        std::strong_ordering operator<=>(const enum64& other) const;
    };

} // mlss
