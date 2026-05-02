#pragma once

namespace mlss
{

    template<std::size_t N, std::size_t NumClasses>
    struct TrainedDecisionTree
    {
        std::array<std::int32_t, N> m_leftChildren;
        std::array<std::int32_t, N> m_rightChildren;
        std::array<float, N> m_thresholds;
        std::array<std::int32_t, N> m_indices;
        std::int32_t m_classes[N][NumClasses];
    };

    template<std::size_t N, std::size_t NumClasses>
    constexpr std::int32_t predictHelper(
        const TrainedDecisionTree<N, NumClasses>& tree,
        std::span<const float> features);

    template<std::size_t N, std::size_t NumClasses>
    constexpr std::int32_t predictHelper(
        const TrainedDecisionTree<N, NumClasses>& tree,
        std::span<const float> features,
        std::span<const std::int32_t> labels);

} // mlss