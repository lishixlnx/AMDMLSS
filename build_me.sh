#!/bin/bash

# Detect host OS so the same script works on Windows (Git Bash / MSYS) and on
# Linux (native or WSL). On Linux we use a different set of CMake presets and
# different compiler defaults; on Windows the original behaviour is preserved.
case "$(uname -s)" in
    Linux*)                             HOST_OS="linux"   ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT*)   HOST_OS="windows" ;;
    *)                                  HOST_OS="unknown" ;;
esac

# Default values (compiler default depends on host OS)
if [[ "$HOST_OS" == "linux" ]]; then
    COMPILER="clang"
else
    COMPILER="clang"
fi
BUILD_TYPE="release"

# HIP_PATH / ROCM_PATH must be supplied by the environment so the build
# adapts to whatever ROCm install the user has (/opt/rocm, /opt/rocm-7.x,
# C:/opt/rocm, a custom prefix, ...). At least one of them must be set;
# if only one is provided we mirror it to the other so downstream CMake
# projects can pick whichever name they prefer.
require_rocm_env() {
    if [[ -z "${HIP_PATH:-}" && -z "${ROCM_PATH:-}" ]]; then
        echo "ERROR: neither HIP_PATH nor ROCM_PATH is set in the environment." >&2
        echo "       Set one of them to your ROCm install root, for example:" >&2
        if [[ "$HOST_OS" == "linux" ]]; then
            echo "           export HIP_PATH=/opt/rocm" >&2
            echo "           export ROCM_PATH=/opt/rocm" >&2
        else
            echo "           export HIP_PATH=C:/opt/rocm   # Git Bash / MSYS" >&2
            echo "           setx   HIP_PATH C:/opt/rocm   # cmd.exe (new shells)" >&2
        fi
        return 1
    fi
    if [[ -z "${HIP_PATH:-}"  ]]; then export HIP_PATH="$ROCM_PATH"; fi
    if [[ -z "${ROCM_PATH:-}" ]]; then export ROCM_PATH="$HIP_PATH"; fi
    return 0
}

require_rocm_env || exit 1

# Function to display usage
usage() {
    echo "Usage: $0 [-c|--compiler <compiler>] [-b|--build <build_type>] [options]"
    if [[ "$HOST_OS" == "linux" ]]; then
        echo "  -c, --compiler           Compiler: clang, gcc, all (default: clang)"
    else
        echo "  -c, --compiler           Compiler: vs2022, vs2026, clang, all (default: clang)"
    fi
    echo "  -b, --build              Build type: debug, release, all (default: release)"
    echo "  --clean-up               Remove all build directories before building"
    echo "  --run-sample-tests       Run sample tests after building them"
    echo "  --no-tests               Library-only build: skip samples, unit tests and mlss-tester"
    echo "  --no-sample-tests        Skip building the samples"
    echo "  --no-unit-tests          Skip building the unit tests (and mlss-tester)"
    echo "  -d, --deploy [path]      Deploy amdmlss and cmake files (default: ./amdmlss_redist)"
    echo ""
    echo "  By default every build also builds the samples and unit tests"
    echo "  (which pulls in mlss-tester). Use --no-tests for a library-only build,"
    echo "  or --no-sample-tests / --no-unit-tests to skip just one."
    echo ""
    echo "  When using -c all, builds with all supported compilers for the host OS."
    echo "  When using -b all, builds both debug and release configurations."
    echo "  Each build will be placed in build/<preset> directory."
    exit 1
}

# Map a user-friendly compiler+config tuple to the actual CMake preset name
# (Linux presets are prefixed with 'linux-' so they coexist with the Windows
# presets in CMakePresets.json).
resolve_preset() {
    local compiler=$1
    local build_type=$2
    if [[ "$HOST_OS" == "linux" ]]; then
        echo "linux-${compiler}-${build_type}"
    else
        echo "${compiler}-${build_type}"
    fi
}

# Function to deploy amdmlss
deploy_amdmlss() {
    local preset=$1
    local deploy_path=$2
    local build_config=${preset##*-}  # Extract build type from preset name
    
    echo "Deploying amdmlss to: $deploy_path"
    
    # Create deployment directory
    mkdir -p "$deploy_path"
    
    # Install using CMake with the correct configuration
    cmake --install "build/${preset}" --config "${build_config^}" --prefix "$deploy_path"
    
    if [ $? -ne 0 ]; then
        echo "Deployment failed for $preset!"
        return 1
    fi
    
    echo "Deployment completed successfully!"
    return 0
}

# Function to build with a single preset
build_single() {
    local preset=$1
    local build_config=${preset##*-}  # Extract build type from preset name
    
    # Samples and unit tests are built on every build by default. --no-tests
    # opts out of both; --no-sample-tests / --no-unit-tests opt out of each
    # one. Only the unit tests link mlss-tester, so the tester is built only
    # when the unit tests are enabled.
    local build_samples="true"
    local build_unit_tests="true"
    if [[ "$NO_TESTS" == "true" || "$NO_SAMPLE_TESTS" == "true" ]]; then
        build_samples="false"
    fi
    if [[ "$NO_TESTS" == "true" || "$NO_UNIT_TESTS" == "true" ]]; then
        build_unit_tests="false"
    fi

    # Build mlss-tester BEFORE the main project so the correct config is installed
    if [[ "$build_unit_tests" == "true" ]]; then
        echo ""
        echo "Building amd-mlss-tester library..."
        local tester_src="${BASH_SOURCE[0]%/*}/3rdparty/amd-mlss-tester"
        local tester_install="${tester_src}/install"

        # Map main-project preset to the matching mlss-tester preset
        local tester_preset=""
        case "$preset" in
            clang-debug)            tester_preset="clang-lib-static-debug"   ;;
            clang-release)          tester_preset="clang-lib-static-release" ;;
            vs2022-*)               tester_preset="vs2022-lib-static"        ;;
            vs2026-*)               tester_preset="vs2026-lib-static"        ;;
            linux-clang-debug)      tester_preset="clang-lib-static-debug"   ;;
            linux-clang-release)    tester_preset="clang-lib-static-release" ;;
            linux-gcc-debug)        tester_preset="gcc-lib-static-debug"     ;;
            linux-gcc-release)      tester_preset="gcc-lib-static-release"   ;;
            *)
                echo "No matching mlss-tester preset for '$preset'!"
                return 1
                ;;
        esac


        local dx12_include_dir="${tester_src}/3rdparty/amd-cross-compiler-tester/lib/include"
        local dx12_source_dir="${tester_src}/3rdparty/amd-cross-compiler-tester/lib/src"

        local tester_config_args=(--preset "$tester_preset"
                                  -DCMAKE_INSTALL_PREFIX="$tester_install"
                                  -DMLSS_ENABLE_HIP=ON
                                  -DMLSS_ENABLE_AOCL=ON
                                  -DMLSS_DX12_INCLUDE_DIR="$dx12_include_dir"
                                  -DMLSS_DX12_SOURCE_DIR="$dx12_source_dir"
                                  -DBUILD_APP=OFF
                                  -DBUILD_TESTING=OFF)

        # On Windows the project must use the ROCm clang++ for HIP source.
        # On Linux the tester's clang-base preset already uses system clang/clang++,
        # which is what we want, so we skip the explicit override there.
        if [[ "$HOST_OS" != "linux" && "$preset" == clang-* ]]; then
            tester_config_args+=("-DCMAKE_CXX_COMPILER=${HIP_PATH}/bin/clang++.exe"
                                 "-DCMAKE_CXX_FLAGS=-Wno-unused-command-line-argument")
        fi

        cmake "${tester_config_args[@]}" -S "$tester_src"
        if [ $? -ne 0 ]; then
            echo "amd-mlss-tester configuration failed!"
            return 1
        fi

        cmake --build "${tester_src}/build/${tester_preset}" --config "${build_config^}"
        if [ $? -ne 0 ]; then
            echo "amd-mlss-tester build failed!"
            return 1
        fi

        cmake --install "${tester_src}/build/${tester_preset}" --config "${build_config^}"
        if [ $? -ne 0 ]; then
            echo "amd-mlss-tester install failed!"
            return 1
        fi
        echo "amd-mlss-tester built and installed successfully!"
    fi

    # Configure with CMake preset.
    #
    # BUILD_TESTS / BUILD_SAMPLES are driven explicitly by this script rather
    # than relying on the CMake defaults. Both default to ON so every build
    # produces the samples and unit tests (the latter linking mlss-tester,
    # built and installed above). The opt-out flags turn each OFF; disabling
    # the unit tests also skips the mlss-tester/AOCL dependency.
    echo ""
    echo "Configuring with CMake preset: $preset..."
    local config_args=(--preset "$preset")
    if [[ "$build_unit_tests" == "true" ]]; then
        config_args+=(-DBUILD_TESTS=ON)
    else
        config_args+=(-DBUILD_TESTS=OFF)
    fi
    if [[ "$build_samples" == "true" ]]; then
        config_args+=(-DBUILD_SAMPLES=ON)
    else
        config_args+=(-DBUILD_SAMPLES=OFF)
    fi
    cmake "${config_args[@]}"

    if [ $? -ne 0 ]; then
        echo "CMake configuration failed for $preset!"
        return 1
    fi

    # Build the project (including unit tests when mlss-tester is available)
    echo "Building project..."
    cmake --build "build/${preset}" --config "${build_config^}"

    if [ $? -ne 0 ]; then
        echo "Build failed for $preset!"
        return 1
    fi

    echo "$preset build completed successfully!"

    if [[ "$build_samples" == "true" ]]; then
        echo "Samples built successfully!"
    fi
    if [[ "$build_unit_tests" == "true" ]]; then
        echo "Unit tests built successfully!"
    fi

    # Run sample tests if requested. The samples are already built as part of
    # the main build above (unless the samples were disabled), so this only
    # executes them.
    if [[ "$RUN_SAMPLE_TESTS" == "true" && "$build_samples" != "true" ]]; then
        echo "--run-sample-tests ignored because the samples were not built."
    elif [[ "$RUN_SAMPLE_TESTS" == "true" ]]; then
        echo ""
        echo "Running sample tests for $preset..."

        # On Linux, sample binaries have no extension. On Windows (Ninja) they
        # are .exe. With MSBuild they live under a Debug/Release subfolder.
        local exe_pattern
        if [[ "$HOST_OS" == "linux" ]]; then
            exe_pattern="*"
        else
            exe_pattern="*.exe"
        fi

        # First try the base samples directory (for Ninja builds)
        local test_dir="build/${preset}/samples"

        # Check whether usable sample files exist in the base directory
        local has_test_files=false
        if [ -d "$test_dir" ]; then
            for f in "$test_dir"/$exe_pattern; do
                if [[ "$HOST_OS" == "linux" ]]; then
                    if [ -f "$f" ] && [ -x "$f" ]; then
                        has_test_files=true
                        break
                    fi
                else
                    if [ -f "$f" ]; then
                        has_test_files=true
                        break
                    fi
                fi
            done
        fi

        # If not found, try with build configuration subdirectory (for MSBuild)
        if [ "$has_test_files" = false ]; then
            test_dir="build/${preset}/samples/${build_config^}"
            if [ ! -d "$test_dir" ]; then
                test_dir="build/${preset}/samples/Debug"
            fi
            if [ ! -d "$test_dir" ]; then
                test_dir="build/${preset}/samples/Release"
            fi
        fi

        if [ -d "$test_dir" ]; then
            local test_failed=0
            for test_exe in "$test_dir"/$exe_pattern; do
                if [[ "$HOST_OS" == "linux" ]]; then
                    [ -f "$test_exe" ] && [ -x "$test_exe" ] || continue
                    local test_name=$(basename "$test_exe")
                else
                    [ -f "$test_exe" ] || continue
                    local test_name=$(basename "$test_exe" .exe)
                fi
                echo "Running test: $test_name"
                "$test_exe"
                if [ $? -ne 0 ]; then
                    echo "Test $test_name failed!"
                    test_failed=1
                else
                    echo "Test $test_name passed!"
                fi
                echo ""
            done

            if [ $test_failed -ne 0 ]; then
                echo "Some tests failed for $preset!"
                return 1
            fi
            echo "All sample tests passed for $preset!"
        else
            echo "No test directory found at $test_dir"
            echo "Skipping sample tests for $preset"
        fi
    fi

    return 0
}

# Parse command line arguments
ARGS_COUNT=$#
BUILD_TYPE_SPECIFIED=false
CLEAN_UP_REQUESTED=false
RUN_SAMPLE_TESTS=false
NO_TESTS=false
NO_SAMPLE_TESTS=false
NO_UNIT_TESTS=false
DEPLOY=false
DEPLOY_PATH="./amdmlss_redist"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_TYPE="$2"
            BUILD_TYPE_SPECIFIED=true
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --clean-up)
            CLEAN_UP_REQUESTED=true
            shift
            ;;
        --run-sample-tests)
            RUN_SAMPLE_TESTS=true
            shift
            ;;
        --no-tests)
            NO_TESTS=true
            shift
            ;;
        --no-sample-tests)
            NO_SAMPLE_TESTS=true
            shift
            ;;
        --no-unit-tests)
            NO_UNIT_TESTS=true
            shift
            ;;
        -d|--deploy)
            DEPLOY=true
            # Check if next argument is a path (doesn't start with -)
            if [[ $# -gt 1 ]] && [[ ! "$2" =~ ^- ]]; then
                DEPLOY_PATH="$2"
                shift 2
            else
                shift
            fi
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Perform clean-up if requested
if [[ "$CLEAN_UP_REQUESTED" == "true" ]]; then
    echo "Cleaning up build directories..."
    if [ -d "build" ]; then
        rm -rf build/
        echo "Removed build directory"
    fi
    if [ -d "build-clang" ]; then
        rm -rf build-clang/
        echo "Removed build-clang directory"
    fi
    if [ -d "build_clang" ]; then
        rm -rf build_clang/
        echo "Removed build_clang directory"
    fi
    if [ -d "build-vs" ]; then
        rm -rf build-vs/
        echo "Removed build-vs directory"
    fi
    if [ -d "build_vs" ]; then
        rm -rf build_vs/
        echo "Removed build_vs directory"
    fi
    if [ -d "buildvsdebug" ]; then
        rm -rf buildvsdebug/
        echo "Removed buildvsdebug directory"
    fi
    echo "Clean-up completed!"
    echo ""
fi

# Initialise submodules to the SHAs pinned by this checkout. We deliberately
# do NOT run "git submodule update --remote": pulling the tip of each tracked
# remote branch mutates the working tree, requires network access, and makes
# builds non-reproducible (and fails in offline/CI environments). Builds use
# the pinned SHAs; bump them explicitly with a separate, intentional commit.
echo "Initialising git submodules..."
git submodule update --init --recursive
if [ $? -ne 0 ]; then
    echo "git submodule update --recursive failed!"
    exit 1
fi
echo ""

# Validate build type
if [[ "$BUILD_TYPE" != "debug" && "$BUILD_TYPE" != "release" && "$BUILD_TYPE" != "all" ]]; then
    echo "Invalid build type: $BUILD_TYPE"
    usage
fi

# Determine which build types to build
if [[ "$BUILD_TYPE" == "all" ]]; then
    BUILD_TYPES=("debug" "release")
    echo "Building both debug and release configurations..."
else
    BUILD_TYPES=("$BUILD_TYPE")
fi

# Compilers supported by the host OS, in build order for "-c all"
if [[ "$HOST_OS" == "linux" ]]; then
    SUPPORTED_COMPILERS=("clang" "gcc")
else
    SUPPORTED_COMPILERS=("vs2026" "vs2022" "clang")
fi

# Handle "all" compiler option
if [[ "$COMPILER" == "all" ]]; then
    echo "Building with all supported compilers for ${HOST_OS}..."
    echo ""

    for build_type in "${BUILD_TYPES[@]}"; do
        for compiler in "${SUPPORTED_COMPILERS[@]}"; do
            preset=$(resolve_preset "$compiler" "$build_type")
            echo "========================================"
            echo "Building with $preset"
            echo "========================================"
            build_single "$preset"
            if [ $? -ne 0 ]; then
                echo "$preset build failed!"
                exit 1
            fi
            echo ""
        done
    done

    echo "All builds completed successfully!"

    # Deploy if requested (use the last built configuration)
    if [[ "$DEPLOY" == "true" ]]; then
        echo ""
        echo "========================================"
        echo "Deploying amdmlss"
        echo "========================================"
        last_compiler="${SUPPORTED_COMPILERS[-1]}"
        last_preset=$(resolve_preset "$last_compiler" "${BUILD_TYPES[-1]}")
        deploy_amdmlss "$last_preset" "$DEPLOY_PATH"
        if [ $? -ne 0 ]; then
            echo "Deployment failed!"
            exit 1
        fi
    fi

    exit 0
fi

# Validate single compiler against the host's supported set
COMPILER_OK=false
for c in "${SUPPORTED_COMPILERS[@]}"; do
    if [[ "$COMPILER" == "$c" ]]; then
        COMPILER_OK=true
        break
    fi
done
if [[ "$COMPILER_OK" != "true" ]]; then
    echo "Invalid compiler '$COMPILER' for host OS '$HOST_OS'."
    echo "Supported on this host: ${SUPPORTED_COMPILERS[*]}, all"
    usage
fi

# Build with single compiler (possibly multiple build types)
if [[ ${#BUILD_TYPES[@]} -gt 1 ]]; then
    echo "Building with compiler: $COMPILER"
    echo ""
fi

LAST_PRESET=""
for build_type in "${BUILD_TYPES[@]}"; do
    PRESET=$(resolve_preset "$COMPILER" "$build_type")

    if [[ ${#BUILD_TYPES[@]} -gt 1 ]]; then
        echo "========================================"
        echo "Building with preset: $PRESET"
        echo "========================================"
    else
        echo "Building with preset: $PRESET"
    fi

    build_single "$PRESET"
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi

    LAST_PRESET="$PRESET"

    if [[ ${#BUILD_TYPES[@]} -gt 1 ]]; then
        echo ""
    fi
done

echo "Build completed successfully!"

# Deploy if requested
if [[ "$DEPLOY" == "true" ]]; then
    echo ""
    echo "========================================"
    echo "Deploying amdmlss"
    echo "========================================"
    deploy_amdmlss "$LAST_PRESET" "$DEPLOY_PATH"
    if [ $? -ne 0 ]; then
        echo "Deployment failed!"
        exit 1
    fi
fi

# If no arguments provided, suggest using -c all
if [ $ARGS_COUNT -eq 0 ]; then
    echo ""
    echo "Tip: Use '$0 -c all' to build with all supported compilers"
fi
