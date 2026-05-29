#include "api_backend.hpp"

// Standard library includes
#include <cstdarg>
#include <deque>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <expected>
#include <algorithm>
#include <iterator>
#include <format>
#include <ranges>
#include <span>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

// Core includes (which includes amdmlss_api_cdefs.h)
#include "core/core.hpp"
#include "core/impl/types/verbose_mode.hxx"

// Include shader headers
#include "shaders/shaders.hpp"
#include "shaders/operators/mha.hpp"
#include "shaders/operators/conv.hpp"
#include "shaders/operators/gemm.hpp"
#include "shaders/operators/gemm_gemm.hpp"
#include "shaders/operators/gqa.hpp"
#include "shaders/operators/mvn.hpp"
#include "shaders/operators/qgemm.hpp"
#include "shaders/operators/rmsnorm.hpp"
#include "shaders/operators/sigmoid_mul.hpp"

namespace mlss
{
    namespace
    {
        //=====================================================================================================================
        template <class T>
        struct NoDeallocation
        {
            constexpr void operator()(T* ptr)
            {
                ptr = nullptr;
            }
        };

        //=====================================================================================================================
        // Convert std::error_code (with MLSSErrorCode) to the appropriate MLSSenum
        MLSSenum errorCodeToMLSSEnum(const std::error_code& ec)
        {
            if (!ec)
            {
                return MLSS_SUCCESS;
            }

            // Check if it's from the MLSS error category
            if (ec.category() == mlss_error_category())
            {
                switch (static_cast<MLSSErrorCode>(ec.value()))
                {
                    case MLSSErrorCode::Success:
                        return MLSS_SUCCESS;
                    case MLSSErrorCode::ShaderInvalidParameters:
                        return MLSS_ERROR_SHADER_INVALID_PARAMETERS;
                    case MLSSErrorCode::ShaderUnsupportedOperator:
                        return MLSS_ERROR_SHADER_UNSUPPORTED_OPERATOR;
                    case MLSSErrorCode::ShaderUnsupportedArchitecture:
                        return MLSS_ERROR_SHADER_UNSUPPORTED_ARCHITECTURE;
                    case MLSSErrorCode::ShaderUnsupportedConfiguration:
                        return MLSS_ERROR_SHADER_UNSUPPORTED_CONFIGURATION;
                    case MLSSErrorCode::ShaderFeatureNotYetImplemented:
                        return MLSS_ERROR_SHADER_FEATURE_NOT_YET_IMPLEMENTED;
                    case MLSSErrorCode::ArchitectureNotSupported:
                        return MLSS_ERROR_ENUM_ARCHITECTURE_NOT_SUPPORTED;
                    case MLSSErrorCode::ArchitectureNotFound:
                        return MLSS_ERROR_ENUM_ARCHITECTURE_NOT_FOUND;
                    case MLSSErrorCode::CodenameNotFound:
                        return MLSS_ERROR_ENUM_CODENAME_NOT_FOUND;
                    default:
                        return MLSS_ERROR_UNKNOWN_ERROR;
                }
            }

            // Unknown error category - return generic failure
            return MLSS_ERROR_FAILURE;
        }

        //=====================================================================================================================
        template <class FunctorType, class InputOutputType, class... Args>
        constexpr MLSSenum createObj(InputOutputType& ref, Args&&... args)
        {
            FunctorType obj;

            // return obj(ptr, std::forward<Args>(args)...).error_or(MLSS_SUCCESS);

            auto tmp = obj(ref, std::forward<Args>(args)...);

            return tmp.error_or(MLSS_SUCCESS);
        }

        //=====================================================================================================================
        template <class T>
        MLSSvoid destroyObj(MLSSvoid* obj)
        {
            if (obj == nullptr)
            {
                return;
            }

            std::default_delete<T> deleter;

            deleter(static_cast<T*>(obj));
        }

        // Invoke fn() and return its result.  On Windows, any structured exception
        // (access violation, illegal instruction, …) raised inside an MLSS
        // getCapsImpl or getBinaries call is caught here and converted to the
        // fallback value so the caller never sees a process crash.
#ifdef _WIN32
        template <class Fn, class Ret = std::invoke_result_t<Fn>>
        Ret seh_call(Fn&& fn, Ret fallback) noexcept
        {
            __try
            {
                return fn();
            }
            __except(GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ||
                     GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION ||
                     GetExceptionCode() == EXCEPTION_STACK_OVERFLOW
                     ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
            {
                return fallback;
            }
        }
#else
        template <class Fn, class Ret = std::invoke_result_t<Fn>>
        Ret seh_call(Fn&& fn, Ret /*fallback*/) noexcept
        {
            return fn();
        }
#endif

        //=====================================================================================================================
        template <class T>
        constexpr auto getShared(MLSSvoid* const ptr)
        {
            return std::shared_ptr<T>(static_cast<T*>(ptr), NoDeallocation<T>());
        }

        //=====================================================================================================================
        template <class T>
        constexpr auto getShared(const MLSSvoid* const ptr)
        {
            return std::shared_ptr<T>(static_cast<T*>(const_cast<MLSSvoid* const>(ptr)), NoDeallocation<T>());
        }

        //=====================================================================================================================
        template <class T>
        constexpr MLSSbool setParams(Context* const context, std::string_view opName, T paramName, const MLSSvoid* const value)
        {

            for (auto& op : context->m_ops)
            {
                if (op.m_op == opName.data())
                {
                    for (auto& param : op.m_params)
                    {
                        if (param.is(paramName))
                        {
                            param = value;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        enum class NotImplementedType
        {
            Warning,
            Error
        };

        template <NotImplementedType err>
        void not_implemented()
        {
            VerboseManager::getInstance().log(std::clog, VerboseLevel::DEBUG) << "Function or Method: " << __func__ << " Not implemented!" << std::endl;

            if constexpr (err == NotImplementedType::Error)
            {
                setLastError(MLSS_ERROR_NOT_IMPLEMENTED);
                throw std::runtime_error("Not implemented!");
            }
            else
            {
                setLastError(MLSS_WARNING_NOT_IMPLEMENTED);
            }
        }

        //=====================================================================================================================
        template <class T>
        T* getTypeFromHandle(MLSShandle handle)
        {
            Any* any_obj = MemoryManager::template getPointer<Any>(handle);

            if (any_obj && anyIs<T>(*any_obj))
            {
                return anyCast<T>(any_obj);
            }

            return nullptr;
        }

        //=====================================================================================================================
        Context* getContextFromHandle(MLSScontext ctx_handle)
        {
            return getTypeFromHandle<Context>(ctx_handle);
        }

        //=====================================================================================================================
        void* getVectorDataByType(const MLSSvector& vec)
        {
            if (vec.m_handle == 0)
            {
                return nullptr;
            }

            Any* any_obj = MemoryManager::template getPointer<Any>(vec.m_handle);
            if (!any_obj || !any_obj->hasValue())
            {
                return nullptr;
            }

            switch (vec.m_type)
            {
                    // Basic integer types
                case MLSS_BOOL:
                case MLSS_UINT8:
                    if (anyIs<std::vector<uint8_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<uint8_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_INT8:
                    if (anyIs<std::vector<int8_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<int8_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_UINT16:
                    if (anyIs<std::vector<uint16_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<uint16_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_INT16:
                    if (anyIs<std::vector<int16_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<int16_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_UINT32:
                    if (anyIs<std::vector<uint32_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<uint32_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_INT32:
                    if (anyIs<std::vector<int32_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<int32_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_UINT64:
                    if (anyIs<std::vector<uint64_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<uint64_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_INT64:
                    if (anyIs<std::vector<int64_t>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<int64_t>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                    // Floating point types
                case MLSS_FLOAT32:
                    if (anyIs<std::vector<float>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<float>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_FLOAT64:
                    if (anyIs<std::vector<double>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<double>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                    // Enum types
                case MLSS_ENUM:
                    if (anyIs<std::vector<MLSSenum>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<MLSSenum>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_ENUM64:
                    if (anyIs<std::vector<enum64>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<enum64>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                    // Structured types
                case MLSS_ARG:
                    if (anyIs<std::vector<MLSSarg>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<MLSSarg>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_DIM3:
                    if (anyIs<std::vector<MLSSdim3>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<MLSSdim3>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_VECTOR:
                    if (anyIs<std::vector<MLSSvector>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<MLSSvector>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_BINARY:
                    if (anyIs<std::vector<MLSSbinary>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<MLSSbinary>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_CONTEXT:
                    if (anyIs<std::vector<Context>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<Context>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                case MLSS_STRING:
                    if (anyIs<std::vector<MLSSstring>>(*any_obj))
                    {
                        auto& storage = anyCast<std::vector<MLSSstring>&>(*any_obj);
                        return storage.data();
                    }
                    break;

                    // Note: The following types might not have direct C++ equivalents or might need special handling:
                    // MLSS_INT4, MLSS_UINT4 - These are typically packed types, not directly supported as standalone types
                    // MLSS_FLOAT4, MLSS_FLOAT8, MLSS_FLOAT16 - These might need special vector types or be represented differently
                    // MLSS_BFLOAT4, MLSS_BFLOAT8, MLSS_BFLOAT16 - Brain floating point formats

                    // Handle unsupported or unknown types
                case MLSS_NONE_TYPE:
                case MLSS_INT4:
                case MLSS_UINT4:
                case MLSS_FLOAT4:
                case MLSS_FLOAT8:
                case MLSS_FLOAT8_FNUZ:
                case MLSS_FLOAT8_OCP:
                case MLSS_FLOAT16:
                case MLSS_BFLOAT4:
                case MLSS_BFLOAT8:
                case MLSS_BFLOAT8_FNUZ:
                case MLSS_BFLOAT8_OCP:
                case MLSS_BFLOAT16:
                case MLSS_UNKNOWN_TYPE:
                case MLSS_CUSTOM_TYPE:
                case MLSS_UNSET_TYPE:
                default:
                    // These types are either not implemented or don't have direct C++ equivalents
                    return nullptr;
            }

            return nullptr;
        }

        const char* getTypeString(MLSSenum type)
        {

            switch (type)
            {
                case MLSS_NONE_TYPE:
                    return "NONE";
                case MLSS_BOOL:
                    return "BOOL";
                case MLSS_INT8:
                    return "INT8";
                case MLSS_UINT8:
                    return "UINT8";
                case MLSS_INT16:
                    return "INT16";
                case MLSS_UINT16:
                    return "UINT16";
                case MLSS_INT32:
                    return "INT32";
                case MLSS_UINT32:
                    return "UINT32";
                case MLSS_INT64:
                    return "INT64";
                case MLSS_UINT64:
                    return "UINT64";
                case MLSS_FLOAT32:
                    return "FLOAT32";
                case MLSS_FLOAT64:
                    return "FLOAT64";
                case MLSS_ENUM:
                    return "ENUM";
                case MLSS_ENUM64:
                    return "ENUM64";
                case MLSS_CONTEXT:
                    return "CONTEXT";
                case MLSS_ARG:
                    return "ARG";
                case MLSS_VECTOR:
                    return "VECTOR";
                case MLSS_DIM3:
                    return "DIM3";
                case MLSS_BINARY:
                    return "BINARY";
                case MLSS_STRING:
                    return "STRING";
                case MLSS_UNKNOWN_TYPE:
                    return "UNKNOWN";
                case MLSS_CUSTOM_TYPE:
                    return "CUSTOM";
                case MLSS_UNSET_TYPE:
                    return "UNSET";
                case MLSS_INT4:
                    return "INT4";
                case MLSS_UINT4:
                    return "UINT4";
                case MLSS_FLOAT4:
                    return "FLOAT4";
                case MLSS_FLOAT8:
                    return "FLOAT8";
                case MLSS_FLOAT8_FNUZ:
                    return "FLOAT8_FNUZ";
                case MLSS_FLOAT8_OCP:
                    return "FLOAT8_OCP";
                case MLSS_FLOAT16:
                    return "FLOAT16";
                case MLSS_BFLOAT4:
                    return "BFLOAT4";
                case MLSS_BFLOAT8:
                    return "BFLOAT8";
                case MLSS_BFLOAT8_FNUZ:
                    return "BFLOAT8_FNUZ";
                case MLSS_BFLOAT8_OCP:
                    return "BFLOAT8_OCP";
                case MLSS_BFLOAT16:
                    return "BFLOAT16";
                default:
                    return "INVALID";
            }
        }

    } // namespace

    //=====================================================================================================================
    MLSSenum lastError = MLSS_SUCCESS;

    //=====================================================================================================================
    MLSSenum setLastError(const MLSSenum& error)
    {
        lastError = error;

        return error;
    }

    //=====================================================================================================================
    MLSSenum resetLastError()
    {
        MLSSenum error = lastError;
        lastError = MLSS_SUCCESS;
        return error;
    }

    //=====================================================================================================================
    MLSSenum returnLastError()
    {
        return lastError;
    }

    //=====================================================================================================================
    struct createContext_t
    {
        using value_type = Context;
        using pointer = value_type*;

        std::expected<MLSSbool, MLSSenum> operator()(MLSScontext& ctx,
                                                     std::string_view asic,
                                                     std::string_view opName,
                                                     va_list* lst) const;
    };

    //=====================================================================================================================
    std::expected<MLSSbool, MLSSenum> createContext_t::operator()(MLSScontext& ctx, std::string_view asic, std::string_view opName, va_list* lst) const
    {
        std::vector<Context::Op> ops;

        ops.emplace_back(Context::Op::create(std::string(opName)));

        if (lst)
        {
            do
            {
                std::string tmp = va_arg(*lst, MLSSstring);

                if (tmp != MLSS_END_LIST)
                {
                    ops.emplace_back(Context::Op::create(tmp));
                }
                else
                {
                    break;
                }
            } while (true);
        }

        Context context_obj(asic, std::move(ops));

        if (context_obj.m_lastError)
        {
            // Return the specific error code from context creation
            MLSSenum specificError = errorCodeToMLSSEnum(context_obj.m_lastError);
            return std::unexpected<MLSSenum>(specificError);
        }

        if (ctx != 0) // ctx contains a valid handle
        {
            // Get the Any from MemoryManager
            Any* existing_any = MemoryManager::template getPointer<Any>(ctx);

            if (existing_any && MemoryManager::isInitialized(&existing_any))
            {
                // Check if Any contains Context
                if (anyIs<Context>(*existing_any))
                {
                    // Update existing Context
                    auto& existing_context = anyCast<Context&>(*existing_any);
                    existing_context.m_asic = context_obj.m_asic;
                    existing_context.m_ops = std::move(context_obj.m_ops);
                    existing_context.m_wasGetCapsCalled = context_obj.m_wasGetCapsCalled;
                }
                else
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_INVALID_PARAMETER);
                }
            }
            else
            {
                return std::unexpected<MLSSenum>(MLSS_ERROR_INVALID_PARAMETER);
            }
        }
        else
        {
            // Store Context directly in Any
            Any context_any = std::move(context_obj);

            // Create new object and handle
            ctx = MemoryManager::addObject(std::move(context_any));

            // Mark the Any as initialized
            Any* new_any = MemoryManager::template getPointer<Any>(ctx);
            if (new_any)
            {
                MemoryManager::markAsInitialized(&new_any);
            }
        }

        return true;
    }

    //=====================================================================================================================
    struct BinaryInfoCollection_t
    {
        // Use std::deque for storage that is read back through raw pointers:
        // unlike std::vector, std::deque does not invalidate references or
        // pointers to existing elements when push_back / emplace_back grows
        // the container. This is required because addString() returns the
        // c_str() of the back element and that pointer must remain valid
        // for every subsequent insertion.
        std::vector<MLSSbinary>             binary_infos;
        std::deque<std::string>             string_storage;
        std::deque<std::vector<MLSSuint32>> constants_storage;

        // Keeps shader Binaries alive for as long as the C-API hands out
        // pointers into Blob::m_pBinary. Without this, dynamically generated
        // binaries (e.g. linked non-relocatable variants) would be freed
        // before the user could read them.
        std::deque<Binaries>                binaries_storage;

        // Helper to add a string and return a MLSSstring (i.e. a char*)
        MLSSstring addString(const std::string& str);

        // Helper to add constants and return handle
        MLSShandle addConstants(const std::vector<MLSSuint32>& constants);
    };

    //=====================================================================================================================
    MLSSstring BinaryInfoCollection_t::addString(const std::string& str)
    {
        string_storage.emplace_back(str);
        return const_cast<char*>(string_storage.back().c_str());
    }

    //=====================================================================================================================
    MLSShandle BinaryInfoCollection_t::addConstants(const std::vector<MLSSuint32>& constants)
    {
        if (constants.empty())
        {
            return 0;
        }
        constants_storage.emplace_back(constants);
        return reinterpret_cast<MLSShandle>(constants_storage.back().data());
    }

    //=====================================================================================================================
    struct createBinaries_t
    {
        using value_type = Binaries;
        using pointer = value_type*;

        // std::unique_ptr<Binaries> operator()(const Context& context, MLSSsize* const n) const;

        std::expected<MLSSbool, MLSSenum> operator()(MLSSbinary*& bin, const MLSScontext& context, MLSSsize* const n) const;
    };

    //=====================================================================================================================
    std::expected<MLSSbool, MLSSenum> createBinaries_t::operator()(MLSSbinary*& bin, const MLSScontext& context, MLSSsize* const n) const
    {
        if (n == nullptr)
        {
            return std::unexpected<MLSSenum>(MLSS_ERROR_INVALID_PARAMETER);
        }

        // Use MemoryManager to get Context from handle
        Context* ctx = getContextFromHandle(context);

        if (!ctx)
        {
            return std::unexpected<MLSSenum>(MLSS_ERROR_INVALID_PARAMETER);
        }

        if (!ctx->m_wasGetCapsCalled)
        {
            MLSSstatus* statuses = nullptr;
            MLSSsize n = 0;

            auto status = getCaps(context, &statuses, &n);

            if (status != MLSS_SUCCESS)
            {
                return status;
            }
        }

        // Create comprehensive collection. binary_infos is the only storage
        // that must be contiguous (for the C-API surface); string_storage
        // and constants_storage are std::deque so growth never invalidates
        // already-handed-out pointers, regardless of how many blobs an op
        // produces.
        BinaryInfoCollection_t collection;
        collection.binary_infos.reserve(ctx->m_ops.size() * 4);

        for (const auto& op : ctx->m_ops)
        {
            Binaries binaries;

            GfxIpTriple gfxArch = IP_GFX_UNKNOWN;
            if (auto gfxIp = architectureStringToGfxIpTriple(ctx->m_asic); gfxIp.has_value())
            {
                gfxArch = gfxIp.value();
            }

            // Use operator classes for all operations
            if (op.m_op == "MLSS_MHA")
            {
                op::OperatorMHA mha_operator;
                mha_operator.setAttributes(op.m_params);
                mha_operator.setGfxIpTriple(gfxArch);

                auto result = mha_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_CONV")
            {
                op::OperatorConv conv_operator;
                conv_operator.setAttributes(op.m_params);
                conv_operator.setGfxIpTriple(gfxArch);

                auto result = conv_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_GEMM")
            {
                op::OperatorGEMM gemm_operator;
                gemm_operator.setAttributes(op.m_params);
                gemm_operator.setGfxIpTriple(gfxArch);

                auto result = gemm_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_GQA")
            {
                op::OperatorGQA gqa_operator;
                gqa_operator.setAttributes(op.m_params);
                gqa_operator.setGfxIpTriple(gfxArch);

                auto result = gqa_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_MVN")
            {
                op::OperatorMVN mvn_operator;
                mvn_operator.setAttributes(op.m_params);
                mvn_operator.setGfxIpTriple(gfxArch);

                auto result = mvn_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_QGEMM")
            {
                op::OperatorQGEMM qgemm_operator;
                qgemm_operator.setAttributes(op.m_params);
                qgemm_operator.setGfxIpTriple(gfxArch);

                auto result = qgemm_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_RMSNORM")
            {
                op::OperatorRmsNorm rmsnorm_operator;
                rmsnorm_operator.setAttributes(op.m_params);
                rmsnorm_operator.setGfxIpTriple(gfxArch);

                auto result = rmsnorm_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_SIGMOID_MUL")
            {
                op::OperatorSigmoidMul sigmoid_mul_operator;
                sigmoid_mul_operator.setAttributes(op.m_params);
                sigmoid_mul_operator.setGfxIpTriple(gfxArch);

                auto result = sigmoid_mul_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else if (op.m_op == "MLSS_GEMMGEMM")
            {
                op::OperatorGemmGemm gemm_gemm_operator;
                gemm_gemm_operator.setAttributes(op.m_params);
                gemm_gemm_operator.setGfxIpTriple(gfxArch);

                auto result = gemm_gemm_operator.getBinaries();
                if (!result.has_value())
                {
                    return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_SUPPORTED);
                }
                binaries = std::move(result.value());
            }
            else
            {
                // Unknown operator
                return std::unexpected<MLSSenum>(MLSS_ERROR_OPERATOR_NOT_FOUND);
            }

            // Helper lambda to create MLSSbinary from a Blob
            auto createBinaryInfo = [&collection, &op, &ctx](const Binaries::Blob& shader_blob) -> MLSSbinary
            {
                MLSSbinary binary_info = {};

                // Store strings with proper lifetime management
                binary_info.m_pOperatorName = collection.addString(op.m_op);
                binary_info.m_ASIC = collection.addString(ctx->m_asic);
                binary_info.m_pKernelName = collection.addString(shader_blob.m_name);

                // Set grid and block dimensions from shader blob
                binary_info.m_grid = shader_blob.m_grid;
                binary_info.m_blocks = shader_blob.m_blocks;
                binary_info.m_sharedMemInBytes = 0;

                // Create constants vector
                if (!shader_blob.m_constants.empty())
                {
                    binary_info.m_constants = createTypedVector<MLSSuint32>(
                        shader_blob.m_constants.data(),
                        shader_blob.m_constants.size());
                }
                else
                {
                    binary_info.m_constants = {};
                    binary_info.m_constants.m_size = 0;
                    binary_info.m_constants.m_type = MLSS_UINT32;
                    binary_info.m_constants.m_handle = 0;
                }

                // Create arguments vector from shader blob
                if (!shader_blob.m_argList.empty())
                {
                    binary_info.m_argList = createTypedVector<MLSSarg>(
                        shader_blob.m_argList.data(),
                        shader_blob.m_argList.size());
                }
                else
                {
                    binary_info.m_argList = {};
                    binary_info.m_argList.m_size = 0;
                    binary_info.m_argList.m_type = MLSS_ARG;
                    binary_info.m_argList.m_handle = 0;
                }

                // Store binary data pointer and size
                binary_info.m_binaries = const_cast<MLSSvoid*>(shader_blob.m_pBinary);
                binary_info.m_binarySize = shader_blob.m_size;

                bool isRelocatable = false;
                if (shader_blob.m_size >= 18u)
                {
                    const auto* raw = static_cast<const std::uint8_t*>(shader_blob.m_pBinary);
                    if (raw[0] == 0x7Fu && raw[1] == 0x45u && raw[2] == 0x4Cu && raw[3] == 0x46u)
                    {
                        std::uint16_t eType = static_cast<std::uint16_t>(raw[16])
                                            | (static_cast<std::uint16_t>(raw[17]) << 8u);
                        isRelocatable = (eType == 1u);
                    }
                }
                binary_info.m_isRelocatable = isRelocatable;

                return binary_info;
            };

            // Keep the Binaries (and the Blobs they own) alive past this
            // loop iteration; the createBinaryInfo lambda above hands out
            // raw pointers into each Blob's underlying buffer.
            collection.binaries_storage.emplace_back(std::move(binaries));
            const auto& storedBinaries = collection.binaries_storage.back();
            for (const auto& blob : storedBinaries)
            {
                collection.binary_infos.emplace_back(createBinaryInfo(blob));
            }
        }

        *n = collection.binary_infos.size();

        // Store the collection in MemoryManager using Any
        Any collection_any = std::move(collection);
        MLSShandle handle = MemoryManager::addObject(std::move(collection_any));

        // Mark as initialized
        Any* new_any = MemoryManager::template getPointer<Any>(handle);
        if (!new_any)
        {
            return std::unexpected<MLSSenum>(MLSS_ERROR_BAD_MEMORY_ALLOCATION);
        }

        MemoryManager::markAsInitialized(&new_any);

        // Get the stored collection to return pointer to its data
        if (anyIs<BinaryInfoCollection_t>(*new_any))
        {
            auto& stored_collection = anyCast<BinaryInfoCollection_t&>(*new_any);
            bin = stored_collection.binary_infos.data();
        }
        else
        {
            return std::unexpected<MLSSenum>(MLSS_ERROR_BAD_MEMORY_ALLOCATION);
        }

        return true;
    }

    //=====================================================================================================================
    MLSSstatus createContext(MLSScontext& context, std::string_view asic, std::string_view opName, va_list* lst)
    {
        return createObj<createContext_t>(context, asic, opName, lst);
    }

    //=====================================================================================================================
    MLSSenum createBinaries(MLSSbinary*& binaries, const MLSScontext context, MLSSsize* const n)
    {
        return createObj<createBinaries_t>(binaries, context, n);
    }

    //=====================================================================================================================
    // Filtered variant: collects all blobs via createBinaries, then compacts the array
    // in-place to keep only entries matching `kind`.
    MLSSenum createBinariesEx(MLSSbinary*& binaries, const MLSScontext context, MLSSsize* const n, MLSSbinaryKind kind)
    {
        MLSSenum status = createBinaries(binaries, context, n);
        if (status != MLSS_SUCCESS)
            return status;
        if (!n || *n == 0 || !binaries)
            return MLSS_SUCCESS;

        if (kind == MLSS_BINARY_KIND_ANY)
            return MLSS_SUCCESS;

        const MLSSsize total = *n;

        // Compact matching binaries into the front of the array so the
        // returned [binaries, binaries + *n) range contains only matches
        // even when matching entries are not contiguous in the original data.
        MLSSsize writeIndex = 0;
        for (MLSSsize i = 0; i < total; ++i)
        {
            if (!binaries[i].m_binaries || binaries[i].m_binarySize == 0)
                continue;

            bool match = (kind == MLSS_BINARY_KIND_NON_RELOCATABLE) ? !binaries[i].m_isRelocatable
                                                                     :  binaries[i].m_isRelocatable;
            if (!match)
                continue;

            if (writeIndex != i)
                binaries[writeIndex] = binaries[i];
            ++writeIndex;
        }

        // If no binaries matched the requested kind, return an empty result
        // set rather than silently exposing the unfiltered binaries.
        *n = writeIndex;
        return MLSS_SUCCESS;
    }

    //=====================================================================================================================
    MLSSbool setParams(MLSScontext* const context, std::string_view opName, std::string_view paramName, const MLSSvoid* const value)
    {
        Context* ctx = getContextFromHandle(*context);
        if (ctx)
        {
            return setParams(ctx, opName, paramName, value);
        }
        return false;
    }

    //=====================================================================================================================
    MLSSbool setParams(MLSScontext* const context, std::string_view opName, const MLSSenum& paramEnum, const MLSSvoid* const value)
    {
        Context* ctx = getContextFromHandle(*context);
        if (ctx)
        {
            return setParams(ctx, opName, paramEnum, value);
        }
        return false;
    }

    //=====================================================================================================================
    MLSSvoid not_implemented_as_a_warning()
    {
        not_implemented<NotImplementedType::Warning>();
    }

    //=====================================================================================================================
    MLSSvoid not_implemented_as_an_error()
    {
        not_implemented<NotImplementedType::Error>();
    }

    //=====================================================================================================================
    MLSSstatus printParams(const MLSScontext context, const std::string& opName)
    {
        // Get Context from handle
        Context* ctx = getContextFromHandle(context);

        if (!ctx)
        {
            error_log << "Error: Invalid context handle" << std::endl;
            return setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        std::cout.sync_with_stdio(true);

        MLSSbool found_operation = false;
        info_log << "" << std::endl;
        info_log << "========================================" << std::endl;
        info_log << "MLSS Parameters for Operation: " << opName << std::endl;
        info_log << "ASIC: " << ctx->m_asic << std::endl;
        info_log << "Total Operations in Context: " << ctx->m_ops.size() << std::endl;
        info_log << "========================================" << std::endl;

        for (const auto& op : ctx->m_ops)
        {
            if (op.m_op == opName)
            {
                found_operation = true;

                if (op.m_params.empty())
                {
                    info_log << "No parameters set for operation '" << opName << "'" << std::endl;
                }
                else
                {
                    info_log << "Parameters (" << op.m_params.size() << " total):" << std::endl;
                    info_log << "----------------------------------------" << std::endl;

                    size_t param_index = 0;
                    for (const auto& param : op.m_params)
                    {
                        try
                        {
                            info_log << "[" << param_index << "] " << param << std::endl;

                            // Add additional parameter metadata
                            info_log << "    Range: " << param.range() << std::endl;
                            info_log << "    Enum: " << param.attr_enum() << std::endl;
                            info_log << "    Type: " << getTypeString(param.element_type_enum()) << std::endl;
                            info_log << "    IsArray: " << (param.isArray() ? "Yes" : "No") << std::endl;
                            if (param.isArray())
                            {
                                info_log << "    Elements: " << param.num_elements() << std::endl;
                            }
                            info_log << "" << std::endl;
                        }
                        catch (const std::exception& e)
                        {
                            error_log << "[" << param_index << "] Error printing parameter '"
                                      << param.name() << "': " << e.what() << std::endl;
                        }
                        param_index++;
                    }
                }
                break; // Found the operation, no need to continue
            }
        }

        if (!found_operation)
        {
            error_log << "Operation '" << opName << "' not found in context." << std::endl;
            info_log << "Available operations:" << std::endl;
            for (size_t i = 0; i < ctx->m_ops.size(); ++i)
            {
                const auto& op = ctx->m_ops[i];
                info_log << "  [" << i << "] " << op.m_op
                         << " (" << op.m_params.size() << " parameters)" << std::endl;
            }
            info_log << "========================================" << std::endl;
            info_log << "" << std::endl;
            return setLastError(MLSS_ERROR_OPERATOR_NOT_FOUND);
        }

        info_log << "========================================" << std::endl;
        info_log << "" << std::endl;

        return setLastError(MLSS_SUCCESS);
    }

    //=====================================================================================================================
    MLSSstatus printBinaries(const MLSSbinary* const binaries, const MLSSsize n)
    {
        if (binaries == nullptr)
        {
            error_log << "Error: binaries pointer is null" << std::endl;
            return setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        if (n == 0)
        {
            info_log << "No binaries to print" << std::endl;
            return setLastError(MLSS_SUCCESS);
        }

        std::cout.sync_with_stdio(true);

        info_log << "" << std::endl;
        info_log << "========================================" << std::endl;
        info_log << "MLSS Binary Information (" << n << " binaries)" << std::endl;
        info_log << "========================================" << std::endl;

        for (MLSSsize i = 0; i < n; ++i)
        {
            const auto& binary = binaries[i];

            info_log << "\n[Binary " << i << "]" << std::endl;
            info_log << "----------------------------------------" << std::endl;

            // Basic information
            info_log << "Operator Name: " << (binary.m_pOperatorName ? binary.m_pOperatorName : "N/A") << std::endl;
            info_log << "ASIC: " << (binary.m_ASIC ? binary.m_ASIC : "N/A") << std::endl;
            info_log << "Kernel Name: " << (binary.m_pKernelName ? binary.m_pKernelName : "N/A") << std::endl;
            info_log << "Relocatable: " << (binary.m_isRelocatable ? "yes" : "no") << std::endl;

            // Grid and block dimensions
            info_log << "Grid: (" << binary.m_grid.m_x << ", " << binary.m_grid.m_y << ", " << binary.m_grid.m_z << ")" << std::endl;
            info_log << "Blocks: (" << binary.m_blocks.m_x << ", " << binary.m_blocks.m_y << ", " << binary.m_blocks.m_z << ")" << std::endl;
            info_log << "Shared Memory: " << binary.m_sharedMemInBytes << " bytes" << std::endl;

            // Constants vector information
            info_log << "Constants Vector:" << std::endl;
            info_log << "  Size: " << binary.m_constants.m_size << std::endl;
            info_log << "  Type: " << getTypeString(binary.m_constants.m_type) << std::endl;
            info_log << "  Handle: 0x" << std::hex << binary.m_constants.m_handle << std::dec << std::endl;

            // Print actual constants data if available
            if (binary.m_constants.m_handle != 0 && binary.m_constants.m_size > 0)
            {
                if (binary.m_constants.m_type == MLSS_UINT32)
                {
                    const uint32_t* constants_data = getVectorData<uint32_t>(binary.m_constants);
                    if (constants_data)
                    {
                        info_log << "  Constants Data: [";
                        for (MLSSsize j = 0; j < std::min(binary.m_constants.m_size, static_cast<MLSSsize>(10)); ++j)
                        {
                            if (j > 0) info_log << ", ";
                            info_log << constants_data[j];
                        }
                        if (binary.m_constants.m_size > 10)
                        {
                            info_log << ", ... (" << (binary.m_constants.m_size - 10) << " more)";
                        }
                        info_log << "]" << std::endl;
                    }
                }
            }

            // Argument list vector information
            info_log << "Argument List Vector:" << std::endl;
            info_log << "  Size: " << binary.m_argList.m_size << std::endl;
            info_log << "  Type: " << getTypeString(binary.m_argList.m_type) << std::endl;
            info_log << "  Handle: 0x" << std::hex << binary.m_argList.m_handle << std::dec << std::endl;

            // Print actual argument data if available
            if (binary.m_argList.m_handle != 0 && binary.m_argList.m_size > 0)
            {
                if (binary.m_argList.m_type == MLSS_ARG)
                {
                    const MLSSarg* args_data = getVectorData<MLSSarg>(binary.m_argList);
                    if (args_data)
                    {
                        info_log << "  Arguments:" << std::endl;

                        const auto args_span = std::span(args_data, static_cast<std::size_t>(binary.m_argList.m_size));
                        for (auto&& [idx, arg] : std::views::enumerate(args_span))
                        {
                            const auto index = static_cast<std::size_t>(idx);
                            const auto* typeCStr = getTypeString(arg.m_type);
                            const std::string_view typeName = typeCStr ? std::string_view(typeCStr) : std::string_view("UNKNOWN");

                            info_log << std::format(
                                            "    [{}] Name: {}, Place: {}, Type: {}, IsPointer: {}, IndirectionLevel: {}, IsConst: {}",
                                            index,
                                            arg.m_name ? arg.m_name : "NULL",
                                            arg.m_place,
                                            typeName,
                                            arg.m_isPointer,
                                            arg.m_indirectionLevel,
                                            arg.m_isConst)
                                     << std::endl;
                        }

                        // Build the complete signature as a string first to avoid multiple INFO: prefixes
                        const auto returnIt = std::ranges::find_if(args_span, [](const MLSSarg& arg)
                                                                   { return arg.m_isReturn; });

                        std::string returnSegment;
                        if (returnIt != args_span.end())
                        {
                            if (returnIt->m_isConst)
                            {
                                returnSegment += "const ";
                            }

                            const auto* typeCStr = getTypeString(returnIt->m_type);
                            std::string_view typeName = typeCStr ? std::string_view(typeCStr) : std::string_view("UNKNOWN");
                            if (typeName == "NONE")
                            {
                                typeName = "VOID";
                            }

                            returnSegment.append(typeName);
                            returnSegment.append(static_cast<std::size_t>(returnIt->m_indirectionLevel), '*');
                        }
                        else
                        {
                            returnSegment = "void";
                        }

                        const std::string_view kernelName = binary.m_pKernelName ? std::string_view(binary.m_pKernelName) : std::string_view("kernel");

                        std::string signature;
                        signature.reserve(128);
                        std::format_to(std::back_inserter(signature), "\n  Function Signature:\n  {} {}(\n", returnSegment, kernelName);

                        bool firstParam = true;
                        for (auto&& [idx, arg] : std::views::enumerate(args_span))
                        {
                            if (arg.m_isReturn)
                            {
                                continue;
                            }

                            if (!firstParam)
                            {
                                std::format_to(std::back_inserter(signature), ",\n");
                            }
                            firstParam = false;

                            const auto* typeCStr = getTypeString(arg.m_type);
                            const std::string_view typeName = typeCStr ? std::string_view(typeCStr) : std::string_view("UNKNOWN");
                            const std::string pointerStars(static_cast<std::size_t>(arg.m_indirectionLevel), '*');

                            std::format_to(
                                std::back_inserter(signature),
                                "    {}{}{} {}",
                                arg.m_isConst ? "const " : "",
                                typeName,
                                pointerStars,
                                arg.m_name ? std::string_view(arg.m_name) : std::string_view("param"));
                        }

                        std::format_to(std::back_inserter(signature), "\n  )");
                        info_log << signature << std::endl;
                        // if (binary.m_argList.m_size > 5)
                        //{
                        //     std::cout << "    ... (" << (binary.m_argList.m_size - 5) << " more arguments)" << std::endl;
                        // }
                    }
                }
            }

            // Binary data information
            info_log << "Binary Data Pointer: " << (binary.m_binaries ? "Valid" : "NULL") << std::endl;
            if (binary.m_binaries)
            {
                info_log << "  Address: 0x" << std::hex << reinterpret_cast<uintptr_t>(binary.m_binaries) << std::dec << std::endl;
                info_log << "  Size: " << binary.m_binarySize << " bytes" << std::endl;
            }
        }

        info_log << "" << std::endl;
        info_log << "========================================" << std::endl;
        info_log << "End of Binary Information" << std::endl;
        info_log << "========================================" << std::endl;
        info_log << "" << std::endl;

        return setLastError(MLSS_SUCCESS);
    }

    //=====================================================================================================================
    MLSSstatus retrieveVectorData(const MLSSvector vector,
                                  MLSSvoid** const data,
                                  MLSSsize* const n,
                                  MLSSenum* const type)
    {
        // Set size if requested
        if (n != nullptr)
        {
            *n = vector.m_size;
        }

        // Set type if requested
        if (type != nullptr)
        {
            *type = vector.m_type;
        }

        // Set data pointer if requested
        if (data != nullptr)
        {
            void* vector_data = getVectorDataByType(vector);
            *data = vector_data;

            if (vector_data == nullptr && vector.m_handle != 0)
            {
                return setLastError(MLSS_ERROR_INVALID_PARAMETER);
            }
        }

        return setLastError(MLSS_SUCCESS);
    }

    //=====================================================================================================================
    MLSSstatus getCaps(const MLSScontext& context, MLSSstatus** const pStatuses, MLSSsize* nStatuses)
    {
        if (pStatuses == nullptr || nStatuses == nullptr)
        {
            return setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        // Get Context from handle
        Context* ctx = getContextFromHandle(context);

        if (!ctx)
        {
            return setLastError(MLSS_ERROR_INVALID_PARAMETER);
        }

        if (ctx->m_wasGetCapsCalled)
        {
            return MLSS_SUCCESS;
        }

        // Create a status collection to store capability check results
        struct CapabilityCollection_t
        {
            std::vector<MLSSstatus> statuses;
            std::vector<std::string> error_messages; // For debugging
        };

        CapabilityCollection_t capability_collection;
        capability_collection.statuses.reserve(ctx->m_ops.size());
        capability_collection.error_messages.reserve(ctx->m_ops.size());

        // Check capabilities for each operation in the context

        MLSSstatus op_status = MLSS_SUCCESS;
        std::string error_msg;

        for (const auto& op : ctx->m_ops)
        {

            try
            {
                GfxIpTriple gfxArch = IP_GFX_UNKNOWN;
                if (auto gfxIp = architectureStringToGfxIpTriple(ctx->m_asic); gfxIp.has_value())
                {
                    gfxArch = gfxIp.value();
                }

                // Use OperatorBase::getCaps for each operator type
                MLSSbool supported = false;

                if (op.m_op == "MLSS_MHA")
                {
                    // Check if architecture is GFX11+ for MHA
                    if (isGfx11Plus(gfxArch))
                    {
                        supported = seh_call([&]{ return op::OperatorMHA::getCaps(op.m_params, gfxArch); }, false);

                        if (supported)
                        {
                            error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                        }
                        else
                        {
                            op_status = MLSS_ERROR_INVALID_PARAMETER;
                            error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                        }
                    }
                    else
                    {
                        op_status = MLSS_ERROR_GRAPHIX_NOT_SUPPORTED;
                        error_msg = "Operation " + op.m_op + " requires GFX11+ architecture";
                        supported = false;
                    }
                }
                else if (op.m_op == "MLSS_CONV")
                {
                    supported = seh_call([&]{ return op::OperatorConv::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_GEMM")
                {
                    supported = seh_call([&]{ return op::OperatorGEMM::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_GQA")
                {
                    supported = seh_call([&]{ return op::OperatorGQA::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_MVN")
                {
                    supported = seh_call([&]{ return op::OperatorMVN::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_QGEMM")
                {
                    supported = seh_call([&]{ return op::OperatorQGEMM::getCaps(op.m_params); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_RMSNORM")
                {
                    supported = seh_call([&]{ return op::OperatorRmsNorm::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_SIGMOID_MUL")
                {
                    supported = seh_call([&]{ return op::OperatorSigmoidMul::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else if (op.m_op == "MLSS_GEMMGEMM")
                {
                    supported = seh_call([&]{ return op::OperatorGemmGemm::getCaps(op.m_params, gfxArch); }, false);

                    if (supported)
                    {
                        error_msg = "Operation " + op.m_op + " is supported on " + ctx->m_asic;
                    }
                    else
                    {
                        op_status = MLSS_ERROR_INVALID_PARAMETER;
                        error_msg = "Operation " + op.m_op + " has invalid parameters for " + ctx->m_asic;
                    }
                }
                else
                {
                    // Unknown operator
                    op_status = MLSS_ERROR_OPERATOR_NOT_FOUND;
                    error_msg = "Unknown operation " + op.m_op;
                    supported = false;
                }
            }
            catch (const std::exception& e)
            {
                op_status = MLSS_ERROR_FAILURE;
                error_msg = "Exception during capability check for " + op.m_op + ": " + e.what();
            }
            catch (...)
            {
                op_status = MLSS_ERROR_FAILURE;
                error_msg = "Unknown exception during capability check for " + op.m_op;
            }

            capability_collection.statuses.push_back(op_status);
            capability_collection.error_messages.push_back(error_msg);
        }

        // Set the output size
        *nStatuses = capability_collection.statuses.size();

        if (capability_collection.statuses.empty())
        {
            *pStatuses = nullptr;
            return setLastError(MLSS_SUCCESS);
        }

        // Store the capability collection in MemoryManager using Any
        Any capability_any = std::move(capability_collection);
        MLSShandle handle = MemoryManager::addObject(std::move(capability_any));

        // Mark as initialized
        Any* new_any = MemoryManager::template getPointer<Any>(handle);
        if (!new_any)
        {
            return setLastError(MLSS_ERROR_BAD_MEMORY_ALLOCATION);
        }

        MemoryManager::markAsInitialized(&new_any);

        // Get the stored collection to return pointer to its data
        if (anyIs<CapabilityCollection_t>(*new_any))
        {
            auto& stored_collection = anyCast<CapabilityCollection_t&>(*new_any);
            *pStatuses = stored_collection.statuses.data();
        }
        else
        {
            return setLastError(MLSS_ERROR_BAD_MEMORY_ALLOCATION);
        }

        if (op_status == MLSS_SUCCESS)
        {
            ctx->m_wasGetCapsCalled = true;
        }

        return setLastError(op_status);
    }

} // namespace mlss
