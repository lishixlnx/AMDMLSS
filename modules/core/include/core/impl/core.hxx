/* Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <atomic>
#include <expected>
#include <filesystem>
#include <stdexcept>
#include <format>
#include <algorithm>
#include <limits>
#include <iostream>
#include <iterator>
#include <functional>
#include <system_error>
#include <span>

#include "amdmlss/amdmlss_api_cdefs.h"

#include "types/enums.hxx"
#include "types/gfxip.hxx"
#include "types/enum64.hxx"
#include "types/context.hxx"
#include "types/binaries.hxx"
#include "types/allocator.hxx"
#include "types/any.hxx"
#include "types/memory.hxx"
#include "types/attribute.hxx"
#include "types/verbose_mode.hxx"

// ML headers
#include "ml/tree.hxx"
#include "ml/tree.inl.hxx"
#include "ml/hypeboloid.hxx"

// Utility headers - organized by category
#include "utils/elf.hxx"
#include "utils/enumUtils.hxx"
#include "utils/adapters.hxx"
#include "utils/compile.hxx"
#include "utils/attributes.hxx"
#include "utils/errors.hxx"
#include "utils/device.hxx"

// Utility implementation headers
#include "utils/enumUtils.inl.hxx"
#include "utils/adapters.inl.hxx"
#include "utils/compile.inl.hxx"
#include "utils/attributes.inl.hxx"

// Type implementation headers (after utils since they depend on them)
#include "types/binaries.inl.hxx"
#include "types/allocator.inl.hxx"
#include "types/any.inl.hxx"
#include "types/memory.inl.hxx"
#include "types/attribute.inl.hxx"
