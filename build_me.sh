#!/bin/bash

# Default values
COMPILER="clang"
BUILD_TYPE="release"

# Function to display usage
usage() {
    echo "Usage: $0 [-c|--compiler <compiler>] [-b|--build <build_type>] [options]"
    echo "  -c, --compiler           Compiler: vs2022, clang, all (default: clang)"
    echo "  -b, --build              Build type: debug, release, all (default: release)"
    echo "  --clean-up               Remove all build directories before building"
    echo "  --build-sample-tests     Build sample tests but don't run them"
    echo "  --run-sample-tests       Build and run sample tests (or just run if already built)"
    echo "  -d, --deploy [path]      Deploy amdmlss and cmake files (default: ./amdmlss_redist)"
    echo ""
    echo "  When using -c all, builds with all supported compilers."
    echo "  When using -b all, builds both debug and release configurations."
    echo "  Each build will be placed in build/<compiler>-<build_type> directory."
    exit 1
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
    
    # Configure with CMake preset
    echo "Configuring with CMake preset: $preset..."
	pwd
    cmake --preset "$preset"
    
    if [ $? -ne 0 ]; then
        echo "CMake configuration failed for $preset!"
        return 1
    fi
    
    # Build the project
    echo "Building project..."
    cmake --build "build/${preset}" --config "${build_config^}"
    
    if [ $? -ne 0 ]; then
        echo "Build failed for $preset!"
        return 1
    fi
    
    echo "$preset build completed successfully!"
    
    # Build sample tests if requested
    if [[ "$BUILD_SAMPLE_TESTS" == "true" ]] || [[ "$RUN_SAMPLE_TESTS" == "true" ]]; then
        echo ""
        echo "Reconfiguring with BUILD_SAMPLES=ON for $preset..."
        cmake --preset "$preset" -DBUILD_SAMPLES=ON
        
        if [ $? -ne 0 ]; then
            echo "CMake reconfiguration failed for $preset!"
            return 1
        fi
        
        echo "Building sample tests for $preset..."
        cmake --build "build/${preset}" --config "${build_config^}"
        
        if [ $? -ne 0 ]; then
            echo "Sample tests build failed for $preset!"
            return 1
        fi
        echo "Sample tests built successfully!"
    fi
    
    # Run sample tests if requested
    if [[ "$RUN_SAMPLE_TESTS" == "true" ]]; then
        echo ""
        echo "Running sample tests for $preset..."
        # First try the base samples directory (for Ninja builds)
        local test_dir="build/${preset}/samples"
        
        # Check if exe files exist in base directory
        local has_exe_files=false
        if [ -d "$test_dir" ]; then
            for f in "$test_dir"/*.exe; do
                if [ -f "$f" ]; then
                    has_exe_files=true
                    break
                fi
            done
        fi
        
        # If not found, try with build configuration subdirectory (for MSBuild)
        if [ "$has_exe_files" = false ]; then
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
            for test_exe in "$test_dir"/*.exe; do
                if [ -f "$test_exe" ]; then
                    local test_name=$(basename "$test_exe" .exe)
                    echo "Running test: $test_name"
                    "$test_exe"
                    if [ $? -ne 0 ]; then
                        echo "Test $test_name failed!"
                        test_failed=1
                    else
                        echo "Test $test_name passed!"
                    fi
                    echo ""
                fi
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
BUILD_SAMPLE_TESTS=false
RUN_SAMPLE_TESTS=false
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
        --build-sample-tests)
            BUILD_SAMPLE_TESTS=true
            shift
            ;;
        --run-sample-tests)
            RUN_SAMPLE_TESTS=true
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

# Handle "all" compiler option
if [[ "$COMPILER" == "all" ]]; then
    echo "Building with all supported compilers..."
    echo ""
    
    # Build all combinations
    for build_type in "${BUILD_TYPES[@]}"; do
        # Build with vs2026
        echo "========================================"
        echo "Building with vs2026-${build_type}"
        echo "========================================"
        build_single "vs2026-${build_type}"
        if [ $? -ne 0 ]; then
            echo "vs2026-${build_type} build failed!"
            exit 1
        fi
        echo ""
        
        # Build with vs2022
        echo "========================================"
        echo "Building with vs2022-${build_type}"
        echo "========================================"
        build_single "vs2022-${build_type}"
        if [ $? -ne 0 ]; then
            echo "vs2022-${build_type} build failed!"
            exit 1
        fi
        echo ""
        
        # Build with clang
        echo "========================================"
        echo "Building with clang-${build_type}"
        echo "========================================"
        build_single "clang-${build_type}"
        if [ $? -ne 0 ]; then
            echo "clang-${build_type} build failed!"
            exit 1
        fi
        echo ""
    done
    
    echo "All builds completed successfully!"
    
    # Deploy if requested (use the last built configuration)
    if [[ "$DEPLOY" == "true" ]]; then
        echo ""
        echo "========================================"
        echo "Deploying amdmlss"
        echo "========================================"
        # Deploy from the last built configuration (clang-release or vs2022-release)
        local last_preset="clang-${BUILD_TYPES[-1]}"
        deploy_amdmlss "$last_preset" "$DEPLOY_PATH"
        if [ $? -ne 0 ]; then
            echo "Deployment failed!"
            exit 1
        fi
    fi
    
    exit 0
fi

# Validate single compiler
if [[ "$COMPILER" != "vs2022" && "$COMPILER" != "vs2026" && "$COMPILER" != "clang" ]]; then
    echo "Invalid compiler: $COMPILER"
    usage
fi

# Build with single compiler (possibly multiple build types)
if [[ ${#BUILD_TYPES[@]} -gt 1 ]]; then
    echo "Building with compiler: $COMPILER"
    echo ""
fi

LAST_PRESET=""
for build_type in "${BUILD_TYPES[@]}"; do
    PRESET="${COMPILER}-${build_type}"
    
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
