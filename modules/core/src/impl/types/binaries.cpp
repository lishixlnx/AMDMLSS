/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "core/core.hpp"

namespace mlss
{

    //=================================================================================================================
    //                                   Binaries
    //=================================================================================================================

    //---------------------------------------------------------------------
    Binaries::Binaries(std::vector<Blob>&& binaries) : m_binaries(std::move(binaries))
    {
    }

    //---------------------------------------------------------------------
    bool Binaries::empty() const
    {
        return m_binaries.empty();
    }

    //---------------------------------------------------------------------
    void Binaries::addBlob(const Blob& blob)
    {
        m_binaries.push_back(blob);
    }

    //---------------------------------------------------------------------
    void Binaries::addBlob(Blob&& blob)
    {
        m_binaries.push_back(std::move(blob));
    }

    //---------------------------------------------------------------------
    void Binaries::addBlob(const std::vector<Blob>& binaries)
    {
        m_binaries.insert(m_binaries.end(), binaries.begin(), binaries.end());
    }

    //---------------------------------------------------------------------
    void Binaries::addBlob(std::vector<Blob>&& binaries)
    {
        m_binaries.insert(m_binaries.end(),
                          std::make_move_iterator(binaries.begin()),
                          std::make_move_iterator(binaries.end()));
    }

    //=================================================================================================================
    //                                   Binaries::Blob
    //=================================================================================================================

    //---------------------------------------------------------------------
    Binaries::Blob::Blob(const void* ptr,
                         const size_t& size,
                         const std::uint32_t& type,
                         const uint32_t& priority,
                         const std::string& name) : m_pBinary(ptr),
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

    //---------------------------------------------------------------------
    void Binaries::Blob::setOwnedBinary(std::vector<std::uint8_t>&& data)
    {
        m_ownedBinary = std::make_shared<std::vector<std::uint8_t>>(std::move(data));
        m_pBinary = m_ownedBinary->data();
        m_size    = m_ownedBinary->size();
    }

} // namespace mlss
