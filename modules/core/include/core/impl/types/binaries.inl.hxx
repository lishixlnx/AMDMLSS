#pragma once

namespace mlss
{
    //=================================================================================================================
    //                                   Binaries::Blob
    //=================================================================================================================

    //---------------------------------------------------------------------
    template <class T, size_t N>
        requires std::is_same_v<T, uint32_t> || std::is_same_v<T, MLSSarg>
    Binaries::Blob& Binaries::Blob::operator=(const std::array<T, N>& obj)
    {
        if constexpr (std::is_same_v<T, uint32_t>)
        {
            m_constants.assign(obj.begin(), obj.end());
        }
        else
        {
            m_argList.assign(obj.begin(), obj.end());
        }

        return (*this);
    }

    //---------------------------------------------------------------------
    template <typename Func, typename... Args>
    void Binaries::Blob::setGrid(const Func& func, const Args&... args)
    {
        // Check if func returns MLSSdim3 (case 1: MLSSdim3 func(Args...))
        if constexpr (std::is_invocable_r_v<MLSSdim3, Func, Args...>)
        {
            m_grid = func(args...);
        }
        // Check if func returns void and takes MLSSdim3& as first param (case 2: void func(MLSSdim3&, Args...))
        else if constexpr (std::is_invocable_v<Func, MLSSdim3&, Args...>)
        {
            func(m_grid, args...);
        }
        else
        {
            static_assert(std::is_invocable_r_v<MLSSdim3, Func, Args...> ||
                              std::is_invocable_v<Func, MLSSdim3&, Args...>,
                          "setGrid requires either: MLSSdim3 func(Args...) or void func(MLSSdim3&, Args...)");
        }
    }

    //---------------------------------------------------------------------
    template <typename Func, typename... Args>
    void Binaries::Blob::setBlocks(const Func& func, const Args&... args)
    {
        // Check if func returns MLSSdim3 (case 1: MLSSdim3 func(Args...))
        if constexpr (std::is_invocable_r_v<MLSSdim3, Func, Args...>)
        {
            m_blocks = func(args...);
        }
        // Check if func returns void and takes MLSSdim3& as first param (case 2: void func(MLSSdim3&, Args...))
        else if constexpr (std::is_invocable_v<Func, MLSSdim3&, Args...>)
        {
            func(m_blocks, args...);
        }
        else
        {
            static_assert(std::is_invocable_r_v<MLSSdim3, Func, Args...> ||
                              std::is_invocable_v<Func, MLSSdim3&, Args...>,
                          "setBlocks requires either: MLSSdim3 func(Args...) or void func(MLSSdim3&, Args...)");
        }
    }

    //---------------------------------------------------------------------
    template <typename Func, typename... Args>
    void Binaries::Blob::setGridBlocks(const Func& func, const Args&... args)
    {
        // Check if func returns std::pair<MLSSdim3, MLSSdim3>
        if constexpr (std::is_invocable_r_v<std::pair<MLSSdim3, MLSSdim3>, Func, Args...>)
        {
            auto [grid, blocks] = func(args...);
            m_grid = grid;
            m_blocks = blocks;
        }
        // Check if func returns void and takes MLSSdim3& references
        else if constexpr (std::is_invocable_v<Func, MLSSdim3&, MLSSdim3&, Args...>)
        {
            func(m_grid, m_blocks, args...);
        }
        else
        {
            static_assert(std::is_invocable_r_v<std::pair<MLSSdim3, MLSSdim3>, Func, Args...> ||
                              std::is_invocable_v<Func, MLSSdim3&, MLSSdim3&, Args...>,
                          "setGridBlocks requires either: std::pair<MLSSdim3, MLSSdim3> func(Args...) or void func(MLSSdim3&, MLSSdim3&, Args...)");
        }
    }

} // namespace mlss
