/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    //=====================================================================================================================
    template<class T>
    Attribute makeAttribute(const std::string& range, const std::uint32_t& attr_enum, const std::uint32_t& type_enum)
    {
        return { T{}, Attribute::AttributeInfo{getAttributeNameFromEnum(attr_enum), range, attr_enum, makeArrayEnum(type_enum)} };
    }

    //=====================================================================================================================
    template<class T>
    Attribute makeAttribute(const std::string& range, const std::uint32_t& attr_enum, const enum64& type_enum)
    {
        return { T{}, Attribute::AttributeInfo{getAttributeNameFromEnum(attr_enum), range, attr_enum, type_enum} };
    }

    //=====================================================================================================================
    template<class CharT, class Traits>
    std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const Attribute& obj)
    {
        os << obj.name() << ": ";

        auto element_type_enum = obj.element_type_enum();

        if (obj.num_elements() > 1)
        {
            auto print_array = [&os, &obj]<typename T>()
            {
                const T* data = obj.data<T>();
                auto n = obj.num_elements();
                if constexpr (std::is_same_v<T, bool>)
                {
                    os << std::boolalpha;
                    std::copy_n(data, n - 1, std::ostream_iterator<bool>(os, ", "));
                    os << data[n - 1];
                    os << std::noboolalpha;
                }
                else if constexpr (std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>)
                {
                    std::transform(data, data + n - 1, std::ostream_iterator<std::int32_t>(os, ", "),
                        [](auto v) { return static_cast<std::int32_t>(v); });
                    os << static_cast<std::int32_t>(data[n - 1]);
                }
                else {
                    std::copy_n(data, n - 1, std::ostream_iterator<T>(os, ", "));
                    os << data[n - 1];
                }
            };


            os << "[";

            switch (element_type_enum)
            {
            case MLSS_BOOL:
                print_array. template operator() < bool > ();
                break;
            case MLSS_INT8:
                print_array. template operator() < std::int8_t > ();
                break;
            case MLSS_UINT8:
                print_array. template operator() < std::uint8_t > ();
                break;
            case MLSS_INT16:
                print_array. template operator() < std::int16_t > ();
                break;
            case MLSS_UINT16:
                print_array. template operator() < std::uint16_t > ();
                break;
            case MLSS_INT32:
                print_array. template operator() < std::int32_t > ();
                break;
            case MLSS_UINT32:
                print_array. template operator() < std::uint32_t > ();
                break;
            case MLSS_INT64:
                print_array. template operator() < std::int64_t > ();
                break;
            case MLSS_UINT64:
                print_array. template operator() < std::uint64_t > ();
                break;
            case MLSS_FLOAT32:
                print_array. template operator() < float > ();
                break;
            case MLSS_FLOAT64:
                print_array. template operator() < double > ();
                break;
            }

            os << "]";
            os << std::endl;
        }
        else
        {
            switch (element_type_enum)
            {
            case MLSS_BOOL:
                os << (obj.value<bool>() ? "true" : "false");
                break;

            case MLSS_INT8:
                os << static_cast<std::int32_t>(obj.value<std::int8_t>());
                break;

            case MLSS_UINT8:
                os << static_cast<std::int32_t>(obj.value<std::uint8_t>());
                break;

            case MLSS_INT16:
                os << obj.value<std::int16_t>();
                break;

            case MLSS_UINT16:
                os << obj.value<std::uint16_t>();
                break;

            case MLSS_INT32:
                os << obj.value<std::int32_t>();
                break;

            case MLSS_UINT32:
                os << obj.value<std::uint32_t>();
                break;

            case MLSS_INT64:
                os << obj.value<std::int64_t>();
                break;

            case MLSS_UINT64:
                os << obj.value<std::uint64_t>();
                break;

            case MLSS_FLOAT32:
                os << obj.value<float>();
                break;

            case MLSS_FLOAT64:
                os << obj.value<double>();
                break;
            }
        }

        return os;
    }

} // namespace mlss

