#pragma once

namespace mlss
{
    // MLSS error codes
    enum class MLSSErrorCode
    {
        Success = 0,
        // Shader errors
        ShaderInvalidParameters,
        ShaderUnsupportedOperator,
        ShaderUnsupportedArchitecture,
        ShaderUnsupportedConfiguration,
        ShaderFeatureNotYetImplemented,
        // Enum lookup errors
        ArchitectureNotSupported,
        ArchitectureNotFound,
        CodenameNotFound
    };

    // Error category for MLSS errors
    class mlss_error_category_impl final : public std::error_category
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
                case MLSSErrorCode::ArchitectureNotSupported:
                    return "Architecture not supported";
                case MLSSErrorCode::ArchitectureNotFound:
                    return "Architecture not found";
                case MLSSErrorCode::CodenameNotFound:
                    return "GPU codename not found";
                default:
                    return "Unknown MLSS error";
            }
        }
    };

    // Get the error category singleton
    inline const std::error_category& mlss_error_category()
    {
        static mlss_error_category_impl instance;
        return instance;
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
    struct is_error_code_enum<mlss::MLSSErrorCode> : true_type
    {
    };
} // namespace std
