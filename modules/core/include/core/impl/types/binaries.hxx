#pragma once

namespace mlss
{    
    //=================================================================================================================
    //                                   Binaries                                                                   
    //=================================================================================================================
    class Binaries
    {
        public:
        struct Blob;

        //---------------------------------------------------------------------
        constexpr Binaries() = default;

        //---------------------------------------------------------------------
        Binaries(std::vector<Blob>&& binaries);

        //---------------------------------------------------------------------
        bool empty() const;

        //---------------------------------------------------------------------
        void addBlob(const Blob& blob);

        //---------------------------------------------------------------------
        void addBlob(Blob&& blob);

        //---------------------------------------------------------------------
        void addBlob(const std::vector<Blob>& binaries);

        //---------------------------------------------------------------------
        void addBlob(std::vector<Blob>&& binaries);

        //---------------------------------------------------------------------
        // Declared here, defined after Blob is complete
        inline const std::vector<Blob>& getBlobs() const;
        inline std::vector<Blob>& getBlobs();
        inline std::size_t size() const;
        inline const Blob& operator[](std::size_t index) const;
        inline Blob& operator[](std::size_t index);
        inline auto begin();
        inline auto end();
        inline auto begin() const;
        inline auto end() const;

        //---------------------------------------------------------------------
        private:
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

    //=================================================================================================================
    //                      Binaries inline member definitions (after Blob is complete)
    //=================================================================================================================
    inline const std::vector<Binaries::Blob>& Binaries::getBlobs() const { return m_binaries; }
    inline std::vector<Binaries::Blob>& Binaries::getBlobs() { return m_binaries; }
    inline std::size_t Binaries::size() const { return m_binaries.size(); }
    inline const Binaries::Blob& Binaries::operator[](std::size_t index) const { return m_binaries[index]; }
    inline Binaries::Blob& Binaries::operator[](std::size_t index) { return m_binaries[index]; }
    inline auto Binaries::begin() { return m_binaries.begin(); }
    inline auto Binaries::end() { return m_binaries.end(); }
    inline auto Binaries::begin() const { return m_binaries.begin(); }
    inline auto Binaries::end() const { return m_binaries.end(); }

} // mlss
