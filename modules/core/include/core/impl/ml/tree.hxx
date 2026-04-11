#pragma once

namespace mlss
{

    template<std::size_t N>
    struct TrainedDecisionTree
    {
        std::array<std::int32_t, N> m_leftChildren;
        std::array<std::int32_t, N> m_rightChildren;
        std::array<float, N> m_thresholds;
        std::array<std::int32_t, N> m_indices; 
        std::array<std::tuple<std::int32_t, std::int32_t>, N> m_classes;
    };

} // mlss