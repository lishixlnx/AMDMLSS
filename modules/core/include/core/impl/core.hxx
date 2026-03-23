#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <type_traits>
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <expected>
#include <filesystem>
#include <stdexcept>
#include <format>
#include <algorithm>
#include <limits>
#include <iostream>
#include <iterator>
#include <functional>


#include "amdmlss/amdmlss_api_cdefs.h"


#include "types/enums.hpp"
#include "types/enum64.hxx"
#include "types/context.hxx"
#include "types/binaries.hxx"
#include "types/allocator.hxx"
#include "types/any.hxx"
#include "types/memory.hxx"
#include "types/attributeInfo.hxx"
#include "types/attribute.hxx"
#include "shader_error_code.hpp"
#include "types/verbose_mode.hxx"

// Utility headers - organized by category
#include "utils/elf.hxx"
#include "utils/enumUtils.hxx"
#include "utils/adapters.hxx"
#include "utils/devices.hxx"
#include "utils/compile.hxx"
#include "utils/attributes.hxx"

// Utility implementation headers
#include "utils/elf.inl.hxx"
#include "utils/enumUtils.inl.hxx"
#include "utils/adapters.inl.hxx"
#include "utils/devices.inl.hxx"
#include "utils/compile.inl.hxx"
#include "utils/attributes.inl.hxx"

// Type implementation headers (after utils since they depend on them)
#include "types/binaries.inl.hxx"
#include "types/allocator.inl.hxx"
#include "types/any.inl.hxx"
#include "types/memory.inl.hxx"
#include "types/attribute.inl.hxx"
