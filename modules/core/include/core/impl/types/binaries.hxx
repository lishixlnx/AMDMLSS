#pragma once

namespace mlss
{    
    //=================================================================================================================
    //                                   Binaries                                                                   
    //=================================================================================================================
    struct Binaries
    {
        struct Blob;

        //---------------------------------------------------------------------
        constexpr Binaries() = default;

        //---------------------------------------------------------------------
        Binaries(std::vector<Blob>&& binaries);

        //---------------------------------------------------------------------
        bool empty() const;

        //---------------------------------------------------------------------
        std::vector<Blob> m_binaries;
    };

    //=================================================================================================================
    //                                   Binaries::Blob                                                                   
    //=================================================================================================================
    struct Binaries::Blob
    {
        //---------------------------------------------------------------------
        Blob() = default;

        //---------------------------------------------------------------------
        Blob(const void* ptr,
               const size_t& size,
               const std::uint32_t& type,
               const uint32_t& priority,
               const std::string& name);

        //---------------------------------------------------------------------
        Blob(const Blob& obj) = default;

        //---------------------------------------------------------------------
        Blob(Blob&& obj) = default;

        //---------------------------------------------------------------------
        ~Blob() = default;

        //---------------------------------------------------------------------
        Blob& operator=(const Blob& obj) = default;

        //---------------------------------------------------------------------
        Blob& operator=(Blob&& obj) = default;

        //---------------------------------------------------------------------
        template<class T, size_t N>
        requires std::is_same_v<T, uint32_t> || std::is_same_v<T, MLSSarg>
        Blob& operator=(const std::array<T, N>& obj);

        void setGrid(const MLSSdim3& grid);
        
        template<typename Func, typename... Args>
        void setGrid(const Func& func, const Args&... args);

        void setBlocks(const MLSSdim3& blocks);
        
        template<typename Func, typename... Args>
        void setBlocks(const Func& func, const Args&... args);

        void setGridBlocks(const MLSSdim3& grid, const MLSSdim3& blocks);

        template<typename Func, typename... Args>
        void setGridBlocks(const Func& func, const Args&... args);

        //---------------------------------------------------------------------
        const void* m_pBinary;
        std::size_t m_size;
        std::uint32_t m_type;
        std::uint32_t m_priority;
        std::vector<std::uint32_t> m_constants;
        std::string m_name;
        std::vector<MLSSarg> m_argList;
        MLSSdim3 m_grid;
        MLSSdim3 m_blocks;
    };


    using Blob = Binaries::Blob;

} // mlss
