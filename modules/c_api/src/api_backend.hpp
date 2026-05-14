#pragma once

// Include C API definitions first
#include "amdmlss/amdmlss_api_cdefs.h"

// Standard library includes
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <cstdarg>

// Include core types
#include "core/core.hpp"
#include "shaders/shaders.hpp"

namespace mlss
{
    //=====================================================================================================================
    MLSSenum setLastError(const MLSSenum& error);

    //=====================================================================================================================
    MLSSenum resetLastError();

    //=====================================================================================================================
    MLSSenum returnLastError();

    //=====================================================================================================================
    MLSSstatus createContext(MLSScontext& context, std::string_view asic, std::string_view opName, va_list* lst);

    //=====================================================================================================================
    MLSSbool createBinaries(MLSSbinary*& binaries, const MLSScontext context, MLSSsize* const n);

    //=====================================================================================================================
    // Filtered variant: returns only blobs matching `kind` (NON_RELOCATABLE, RELOCATABLE, or ANY).
    MLSSenum createBinariesEx(MLSSbinary*& binaries, const MLSScontext context, MLSSsize* const n, MLSSbinaryKind kind);

    //=====================================================================================================================
    MLSSbool setParams(MLSScontext* const context, std::string_view opName, std::string_view paramName, const MLSSvoid* const value);

    //=====================================================================================================================
    MLSSbool setParams(MLSScontext* const context, std::string_view opName, const MLSSenum& paramEnum, const MLSSvoid* const value);

    //=====================================================================================================================
    MLSSvoid not_implemented_as_a_warning();

    //=====================================================================================================================
    MLSSvoid not_implemented_as_an_error();

    MLSSstatus printParams(const MLSScontext context, const std::string& opName);

    MLSSstatus printBinaries(const MLSSbinary* const binaries, const MLSSsize n);

    MLSSstatus retrieveVectorData(const MLSSvector vector,
                                  MLSSvoid** const data,
                                  MLSSsize* const n,
                                  MLSSenum* const type);

    MLSSstatus getCaps(const MLSScontext& context, MLSSstatus** const pStatuses, MLSSsize* const nStatuses);

} // namespace mlss
