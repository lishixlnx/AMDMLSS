/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

namespace mlss
{

    template<std::size_t N, std::size_t NumClasses>
    constexpr std::int32_t predictHelper(
        const TrainedDecisionTree<N, NumClasses>& tree,
        std::span<const float> features)
    {
        std::int32_t node = 0;

        while (tree.m_leftChildren[node] != -1)
        {
            if (features[tree.m_indices[node]] <= tree.m_thresholds[node])
            {
                node = tree.m_leftChildren[node];
            }
            else
            {
                node = tree.m_rightChildren[node];
            }
        }

        const auto& classDistrib = tree.m_classes[node];
        std::int32_t bestClass = 0;
        std::int32_t bestCount = classDistrib[0];

        for (std::size_t i = 1; i < NumClasses; ++i)
        {
            if (classDistrib[i] > bestCount)
            {
                bestCount = classDistrib[i];
                bestClass = static_cast<std::int32_t>(i);
            }
        }

        return bestClass;
    }

    template<std::size_t N, std::size_t NumClasses>
    constexpr std::int32_t predictHelper(
        const TrainedDecisionTree<N, NumClasses>& tree,
        std::span<const float> features,
        std::span<const std::int32_t> labels)
    {
        auto classIdx = predictHelper(tree, features);
        return labels[classIdx];
    }

} // namespace mlss
