#include "core/core.hpp"

namespace mlss
{
    //=====================================================================================================================
    //                                   Context
    //=====================================================================================================================

    //---------------------------------------------------------------------
    Context::Context(std::string_view asic, std::vector<Op>&& ops) : m_asic(asic),
                                                                     m_ops(std::move(ops))
    {
        if (m_asic == MLSS_GFXAUTOFIND)
        {
            if (auto optimalDevice = getOptimalDeviceFeatures(); optimalDevice.has_value())
            {
                const auto& deviceFeatures = optimalDevice.value();
                m_asic = deviceFeatures.m_gfx;
            }
        }

        // Convert string to GfxIpTriple and refine to high-end variant
        if (auto gfxIp = architectureStringToGfxIpTriple(m_asic); gfxIp.has_value())
        {
            if (auto highEndArch = getHighEndGfxIpTriple(gfxIp.value()); highEndArch.has_value())
            {
                if (auto archString = gfxIpTripleToString(highEndArch.value()); archString.has_value())
                {
                    m_asic = archString.value();
                }
                else
                {
                    m_lastError = archString.error();
                }
            }
            else
            {
                m_lastError = highEndArch.error();
            }
        }
        else
        {
            m_lastError = gfxIp.error();
        }
    }

    //---------------------------------------------------------------------
    bool Context::empty() const
    {
        return m_ops.empty();
    }

    //=====================================================================================================================
    //                                   Context::Op
    //=====================================================================================================================

    //---------------------------------------------------------------------
    Context::Op Context::Op::create(const std::string& opName)
    {
        Context::Op op{opName, {}};

        createAttributes(opName, op.m_params);

        return std::move(op);
    }
} // namespace mlss
