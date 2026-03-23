#pragma once

#include <system_error>

namespace mlss
{
    // Shader-specific error codes
    enum class ShaderErrorCode
    {
        Success = 0,
        InvalidParameters,
        UnsupportedOperator,
        UnsupportedArchitecture,
        UnsupportedConfiguration,
        FeatureNotYetImplemented
    };

    // MLSS error codes (alias for compatibility)
    enum class MLSSErrorCode
    {
        Success = 0,
        ShaderInvalidParameters,
        ShaderUnsupportedOperator,
        ShaderUnsupportedArchitecture,
        ShaderUnsupportedConfiguration,
        ShaderFeatureNotYetImplemented
    };

    // Error category for shader errors
    class shader_error_category_impl : public std::error_category
    {
    public:
        const char* name() const noexcept override
        {
            return "shader";
        }

        std::string message(int ev) const override
        {
            switch (static_cast<ShaderErrorCode>(ev))
            {
            case ShaderErrorCode::Success:
                return "Success";
            case ShaderErrorCode::InvalidParameters:
                return "Invalid parameters";
            case ShaderErrorCode::UnsupportedOperator:
                return "Unsupported operator";
            case ShaderErrorCode::UnsupportedArchitecture:
                return "Unsupported architecture";
            case ShaderErrorCode::UnsupportedConfiguration:
                return "Unsupported configuration";
            case ShaderErrorCode::FeatureNotYetImplemented:
                return "Feature not yet implemented";
            default:
                return "Unknown shader error";
            }
        }
    };

    // Error category for MLSS errors
    class mlss_error_category_impl : public std::error_category
    {
    public:
        const char* name() const noexcept override
        {
            return "mlss";
        }

        std::string message(int ev) const override
        {
            switch (static_cast<MLSSErrorCode>(ev))
            {
            case MLSSErrorCode::Success:
                return "Success";
            case MLSSErrorCode::ShaderInvalidParameters:
                return "Shader invalid parameters";
            case MLSSErrorCode::ShaderUnsupportedOperator:
                return "Shader unsupported operator";
            case MLSSErrorCode::ShaderUnsupportedArchitecture:
                return "Shader unsupported architecture";
            case MLSSErrorCode::ShaderUnsupportedConfiguration:
                return "Shader unsupported configuration";
            case MLSSErrorCode::ShaderFeatureNotYetImplemented:
                return "Shader feature not yet implemented";
            default:
                return "Unknown MLSS error";
            }
        }
    };

    // Get the error category singletons
    inline const std::error_category& shader_error_category()
    {
        static shader_error_category_impl instance;
        return instance;
    }

    inline const std::error_category& mlss_error_category()
    {
        static mlss_error_category_impl instance;
        return instance;
    }

    // Make error_code from ShaderErrorCode
    inline std::error_code make_error_code(ShaderErrorCode e)
    {
        return std::error_code(static_cast<int>(e), shader_error_category());
    }

    // Make error_code from MLSSErrorCode
    inline std::error_code make_error_code(MLSSErrorCode e)
    {
        return std::error_code(static_cast<int>(e), mlss_error_category());
    }

} // namespace mlss

// Register with std::error_code system
namespace std
{
    template <>
    struct is_error_code_enum<mlss::ShaderErrorCode> : true_type {};

    template <>
    struct is_error_code_enum<mlss::MLSSErrorCode> : true_type {};
}
