/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{
    class Attribute;

    //=================================================================================================================
    //                                   Context
    //=================================================================================================================
    struct Context
    {
        struct Op;

        //---------------------------------------------------------------------
        // Not constexpr: std::error_code's default constructor is not constexpr
        // in libstdc++ (it references system_category()), and Context holds
        // std::string/std::vector members, so it can never be a constant
        // expression regardless.
        Context() = default;

        //---------------------------------------------------------------------
        Context(std::string_view asic, std::vector<Op>&& ops);

        //---------------------------------------------------------------------
        bool empty() const;

        //---------------------------------------------------------------------
        std::string m_asic;
        std::vector<Op> m_ops;
        std::error_code m_lastError;
        bool m_wasGetCapsCalled = false;
    };

    //=================================================================================================================
    //                                   Context::Op
    //=================================================================================================================
    struct Context::Op
    {
        //---------------------------------------------------------------------
        static Op create(const std::string& opName);

        //---------------------------------------------------------------------
        std::string m_op;
        std::vector<Attribute> m_params;
    };

} // namespace mlss
