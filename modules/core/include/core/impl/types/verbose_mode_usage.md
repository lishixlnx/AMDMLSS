# Verbose Mode Usage Guide

## Overview

The verbose mode interface provides conditional logging functionality for the AMDMLSS library. It allows controlling the verbosity of output from functions outside of print functions, making debugging and monitoring easier while keeping production logs clean.

## Features

- **Multiple Verbosity Levels**: NONE (0), ERROR (1), WARNING (2), INFO (3), DEBUG (4), TRACE (5)
- **Thread-Safe**: Uses mutex protection for concurrent access
- **Environment Variable Support**: Can be configured via `MLSS_VERBOSE` environment variable
- **Static Logger Instances**: Pre-configured loggers for different levels
- **Custom endl() Method**: Avoids issues with `std::endl`
- **Stream Manipulator Support**: Full support for `std::endl`, `std::flush`, and `std::ends`
- **Automatic Level Prefixes**: Each log line is prefixed with its level (ERROR:, WARNING:, etc.)

## C++ API Usage

### Include Header
```cpp
#include "core/impl/types/verbose_mode.hxx"
```

### Using Static Logger Instances

The verbose mode supports two syntaxes:

#### 1. Original Syntax with Parentheses (Recommended)
```cpp
// Error messages (output to std::cerr)
error_log() << "An error occurred: " << errorCode << std::endl;

// Warning messages (output to std::cerr)
warning_log() << "Warning: deprecated function used" << std::endl;

// Info messages (output to std::cout)
info_log() << "Processing started for " << filename << std::endl;

// Debug messages (output to std::clog)
debug_log() << "Variable value: " << value << std::endl;

// Trace messages (output to std::clog)
trace_log() << "Entering function: " << __func__ << std::endl;
```

#### 2. Alternative Syntax without Parentheses
```cpp
// The VerboseLoggerProxy class also supports this syntax:
info_log << "This is an info message" << std::endl;
warning_log << "This is a warning" << std::endl;

// Note: This syntax creates a new ConditionalStream for each use,
// which may have slightly different behavior than the parentheses syntax
```

Both syntaxes support:
- Stream manipulators (`std::endl`, `std::flush`, `std::ends`)
- Chaining multiple values
- Automatic level prefix insertion (ERROR:, WARNING:, INFO:, etc.)
- The `flush()` method for explicit flushing

### Setting Verbose Level Programmatically
```cpp
// Set verbose level
mlss::VerboseManager::getInstance().setLevel(mlss::VerboseLevel::DEBUG);

// Get current verbose level
mlss::VerboseLevel currentLevel = mlss::VerboseManager::getInstance().getLevel();
```

## C API Usage

### Include Header
```c
#include <amdmlss/amdmlss_api.h>
```

### Available Functions
```c
// Set verbose level (0-5)
mlss_status_t mlssSetVerboseLevel(mlss_enum_t level);

// Get current verbose level
mlss_enum_t mlssGetVerboseLevel();

// Enable verbose mode (sets to INFO level)
mlss_status_t mlssEnableVerboseMode();

// Disable verbose mode (sets to NONE level)
mlss_status_t mlssDisableVerboseMode();
```

### Example Usage
```c
// Enable verbose mode at INFO level
mlssSetVerboseLevel(3);

// Your MLSS operations will now output INFO, WARNING, and ERROR messages
mlssPrintParameters(context, opName);
mlssGetBinaries(context, &binaries, &n);

// Change to DEBUG level for more detailed output
mlssSetVerboseLevel(4);

// Disable verbose mode
mlssDisableVerboseMode();
```


## Verbose Levels

| Level | Value | Description | Output Stream |
|-------|-------|-------------|---------------|
| NONE | 0 | No output | - |
| ERROR | 1 | Only error messages | std::cerr |
| WARNING | 2 | Errors and warnings | std::cerr |
| INFO | 3 | Errors, warnings, and info | std::cout/std::cerr |
| DEBUG | 4 | All above plus debug info | std::clog |
| TRACE | 5 | All messages including trace | std::clog |

## Implementation Notes

1. **No std::endl**: The implementation uses a custom `endl()` method instead of `std::endl` to avoid flushing issues
2. **Static Instantiation**: Logger instances are statically created, avoiding macro usage
3. **Namespace**: All verbose mode functionality is in the `mlss` namespace
4. **Thread Safety**: The VerboseManager singleton is thread-safe using std::mutex
5. **Conditional Compilation**: The verbose mode is always available, not conditionally compiled
6. **Stream Manipulators**: Full support for `std::endl`, `std::flush`, and `std::ends` through operator overloading
7. **Inline Linkage**: Logger instances use `inline` keyword to ensure proper linkage across translation units

## Example: MHA Verbose Mode Sample

See `samples/mha_verbose_example.c` for a complete example demonstrating:
- Enabling verbose mode at different levels
- Viewing verbose output from MLSS operations
- Switching between verbose levels
- Disabling and re-enabling verbose mode
