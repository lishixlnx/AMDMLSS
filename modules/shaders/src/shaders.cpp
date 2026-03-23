#include "shaders/shaders.hpp"

// Note: shaderCallHipMha.hpp has been removed
// #include "shaders/interface/shaderCallHipMha.hpp"

#include <algorithm>
#include <cctype>
#include <iterator>


namespace mlss::shaders
{




    //=====================================================================================================================
    std::uint32_t convertShaderErrorToEnum(std::error_code error)
    {
        if (error.category() == shader_error_category())
        {
            switch (static_cast<ShaderErrorCode>(error.value()))
            {
            case ShaderErrorCode::InvalidParameters:
                return MLSS_ERROR_INVALID_PARAMETER;
            case ShaderErrorCode::UnsupportedOperator:
                return MLSS_ERROR_OPERATOR_NOT_SUPPORTED;
            case ShaderErrorCode::UnsupportedArchitecture:
                return MLSS_ERROR_GRAPHIX_NOT_SUPPORTED;
            case ShaderErrorCode::FeatureNotYetImplemented:
                return MLSS_ERROR_NOT_IMPLEMENTED;
            default:
                return MLSS_ERROR_FAILURE;
            }
        }
        return MLSS_ERROR_FAILURE;
    }

} // mlss::shaders
