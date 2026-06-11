/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#include "core/core.hpp"

namespace mlss
{

    std::mutex VerboseManager::ConditionalStream::mutex_;

    VerboseManager::VerboseManager()
        : level_(VerboseLevel::DEBUG)
    {
    }

    VerboseManager::ConditionalStream::ConditionalStream(std::ostream& stream, bool enabled, VerboseLevel level)
        : stream_(stream), enabled_(enabled), level_(level), prefix_added_(false)
    {
    }

    VerboseManager::ConditionalStream::ConditionalStream(ConditionalStream&& other) noexcept
        : stream_(other.stream_), enabled_(other.enabled_), level_(other.level_), buffer_(std::move(other.buffer_)), prefix_added_(other.prefix_added_)
    {
        other.enabled_ = false;
    }

    VerboseManager::ConditionalStream& VerboseManager::ConditionalStream::operator=(ConditionalStream&& other) noexcept
    {
        if (this != &other)
        {
            stream_ = other.stream_;
            enabled_ = other.enabled_;
            level_ = other.level_;
            buffer_ = std::move(other.buffer_);
            prefix_added_ = other.prefix_added_;
            other.enabled_ = false;
        }
        return *this;
    }

    VerboseManager::ConditionalStream::~ConditionalStream()
    {
        if (enabled_ && buffer_.tellp() > 0)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!prefix_added_)
            {
                stream_.get() << getLevelPrefix();
            }
            stream_.get() << buffer_.str();
            stream_.get().flush();
        }
    }

    void VerboseManager::ConditionalStream::addPrefix()
    {
        if (!prefix_added_ && enabled_)
        {
            buffer_ << getLevelPrefix();
            prefix_added_ = true;
        }
    }

    const char* VerboseManager::ConditionalStream::getLevelPrefix() const
    {
        switch (level_)
        {
            case VerboseLevel::ERROR:
                return "ERROR: ";
            case VerboseLevel::WARNING:
                return "WARNING: ";
            case VerboseLevel::INFO:
                return "INFO: ";
            case VerboseLevel::DEBUG:
                return "DEBUG: ";
            case VerboseLevel::TRACE:
                return "TRACE: ";
            default:
                return "";
        }
    }

    VerboseManager::ConditionalStream& VerboseManager::ConditionalStream::operator<<(std::ostream& (*pf)(std::ostream&))
    {
        if (enabled_)
        {
            if (pf == static_cast<std::ostream& (*)(std::ostream&)>(std::endl))
            {
                addPrefix();
                buffer_ << '\n';
                // Flush the buffer
                std::lock_guard<std::mutex> lock(mutex_);
                stream_.get() << buffer_.str();
                stream_.get().flush();
                buffer_.str("");
                buffer_.clear();
                prefix_added_ = false;
            }
            else if (pf == static_cast<std::ostream& (*)(std::ostream&)>(std::flush))
            {
                addPrefix();
                // Just flush without newline
                std::lock_guard<std::mutex> lock(mutex_);
                stream_.get() << buffer_.str();
                stream_.get().flush();
                buffer_.str("");
                buffer_.clear();
            }
            else if (pf == static_cast<std::ostream& (*)(std::ostream&)>(std::ends))
            {
                addPrefix();
                // Add null character
                buffer_ << '\0';
            }
        }
        return *this;
    }

    VerboseManager::ConditionalStream& VerboseManager::ConditionalStream::endl()
    {
        if (enabled_)
        {
            addPrefix();
            buffer_ << '\n';
        }
        return *this;
    }

    VerboseManager::ConditionalStream& VerboseManager::ConditionalStream::flush()
    {
        if (enabled_ && buffer_.tellp() > 0)
        {
            addPrefix();
            std::lock_guard<std::mutex> lock(mutex_);
            stream_.get() << buffer_.str();
            stream_.get().flush();
            buffer_.str("");
            buffer_.clear();
        }
        return *this;
    }

} // namespace mlss
