#pragma once

#include <iostream>
#include <sstream>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <functional>

namespace mlss
{

enum class VerboseLevel : int
{
    NONE = 0,
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5
};

class VerboseManager
{
public:
    static VerboseManager& getInstance()
    {
        static VerboseManager instance;
        return instance;
    }

    void setLevel(VerboseLevel level)
    {
        level_.store(level);
    }

    VerboseLevel getLevel() const
    {
        return level_.load();
    }

    bool isEnabled(VerboseLevel level) const
    {
        return static_cast<int>(level) <= static_cast<int>(level_.load());
    }

    class ConditionalStream
    {
    public:
        ConditionalStream(std::ostream& stream, bool enabled, VerboseLevel level);
        ~ConditionalStream();

        // Delete copy constructor and copy assignment
        ConditionalStream(const ConditionalStream&) = delete;
        ConditionalStream& operator=(const ConditionalStream&) = delete;

        // Move constructor and move assignment
        ConditionalStream(ConditionalStream&& other) noexcept;
        ConditionalStream& operator=(ConditionalStream&& other) noexcept;

        template<typename T>
        ConditionalStream& operator<<(const T& value)
        {
            if (enabled_)
            {
                if (!prefix_added_ && buffer_.tellp() == 0)
                {
                    addPrefix();
                }
                buffer_ << value;
            }
            return *this;
        }

        // Support for std::endl and std::flush
        ConditionalStream& operator<<(std::ostream& (*pf)(std::ostream&));

        ConditionalStream& endl();
        ConditionalStream& flush();

    private:
        std::reference_wrapper<std::ostream> stream_;
        bool enabled_;
        VerboseLevel level_;
        std::ostringstream buffer_;
        static std::mutex mutex_;
        bool prefix_added_;

        void addPrefix();
        const char* getLevelPrefix() const;
    };

    ConditionalStream log(std::ostream& stream, VerboseLevel level)
    {
        return ConditionalStream(stream, isEnabled(level), level);
    }

private:
    VerboseManager();
    VerboseManager(const VerboseManager&) = delete;
    VerboseManager& operator=(const VerboseManager&) = delete;

    std::atomic<VerboseLevel> level_;
};

// Proxy class that creates ConditionalStream on demand
class VerboseLoggerProxy
{
public:
    VerboseLoggerProxy(std::ostream& stream, VerboseLevel level)
        : stream_(stream), level_(level) {}

    // Conversion operator to create ConditionalStream when used
    operator VerboseManager::ConditionalStream()
    {
        return VerboseManager::getInstance().log(stream_, level_);
    }

    // Template operator<< to handle any type
    template<typename T>
    VerboseManager::ConditionalStream operator<<(const T& value)
    {
        auto stream = VerboseManager::getInstance().log(stream_, level_);
        stream << value;
        return std::move(stream);
    }

    // Special handling for stream manipulators
    VerboseManager::ConditionalStream operator<<(std::ostream& (*pf)(std::ostream&))
    {
        auto stream = VerboseManager::getInstance().log(stream_, level_);
        stream << pf;
        return std::move(stream);
    }

private:
    std::ostream& stream_;
    VerboseLevel level_;
};

// Static logger instances - now can be used directly without ()
// Using inline to ensure external linkage and single definition across translation units
inline VerboseLoggerProxy error_log(std::cerr, VerboseLevel::ERROR);
inline VerboseLoggerProxy warning_log(std::cerr, VerboseLevel::WARNING);
inline VerboseLoggerProxy info_log(std::cout, VerboseLevel::INFO);
inline VerboseLoggerProxy debug_log(std::clog, VerboseLevel::DEBUG);
inline VerboseLoggerProxy trace_log(std::clog, VerboseLevel::TRACE);

} // namespace mlss
