#include "core/core.hpp"

namespace mlss
{

    //=================================================================================================================
    //                                   Binaries                                                                   
    //=================================================================================================================

//---------------------------------------------------------------------
    Binaries::Binaries(std::vector<Blob>&& binaries) :
        m_binaries(std::move(binaries))
    {
    }

    //---------------------------------------------------------------------
    bool Binaries::empty() const
    {
        return m_binaries.empty();
    }

    //=================================================================================================================
    //                                   Binaries::Blob                                                                   
    //=================================================================================================================

    //---------------------------------------------------------------------
    Binaries::Blob::Blob(const void* ptr,
        const size_t& size,
        const std::uint32_t& type,
        const uint32_t& priority,
        const std::string& name) :
        m_pBinary(ptr),
        m_size(size),
        m_type(type),
        m_priority(priority),
        m_name(name)
    {
    }

    //---------------------------------------------------------------------
    void Binaries::Blob::setGrid(const MLSSdim3& grid)
    {
        m_grid = grid;
    }

    //---------------------------------------------------------------------
    void Binaries::Blob::setBlocks(const MLSSdim3& blocks)
    {
        m_blocks = blocks;
    }

    //---------------------------------------------------------------------
    void Binaries::Blob::setGridBlocks(const MLSSdim3& grid, const MLSSdim3& blocks)
    {
        m_grid = grid;
        m_blocks = blocks;
    }

} // mlss
