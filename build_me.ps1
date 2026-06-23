# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
[CmdletBinding()]
param(
    [Parameter()]
    [Alias('c')]
    [ValidateSet('vs2022', 'vs2026', 'clang', 'all')]
    [string]$Compiler = 'clang',

    [Parameter()]
    [Alias('b')]
    [ValidateSet('debug', 'release', 'all')]
    [string]$Build = 'release',

    [Parameter()]
    [switch]$Help,

    [Parameter()]
    [switch]$CleanUp,

    [Parameter()]
    [switch]$RunSampleTests,

    # Samples and unit tests (and therefore mlss-tester) are built on every
    # build by default. -NoTests opts out of both (library-only build);
    # -NoSampleTests and -NoUnitTests opt out of each individually.
    [Parameter()]
    [switch]$NoTests,

    [Parameter()]
    [switch]$NoSampleTests,

    [Parameter()]
    [switch]$NoUnitTests,

    # PowerShell cannot bind a string parameter that was passed without a value
    # (e.g. '-d'), so deployment is split into a switch (enable) and a separate
    # path argument. Backwards-compat helper for the legacy '-d <path>' form is
    # provided below by detecting a string positional after -d/--Deploy.
    [Parameter()]
    [Alias('d')]
    [switch]$Deploy,

    [Parameter()]
    [Alias('Path')]
    [string]$DeployPath = './amdmlss_redist',

    [Parameter()]
    [switch]$h
)

# Handle short form help
if ($h) {
    $Help = $true
}

# Display help if requested
if ($Help) {
    Write-Host "Usage: .\build_me.ps1 [-c|-Compiler <compiler>] [-b|-Build <build_type>] [options]"
    Write-Host "  -c, -Compiler           Compiler: vs2022, vs2026, clang, all (default: clang)"
    Write-Host "  -b, -Build              Build type: debug, release, all (default: release)"
    Write-Host "  -CleanUp                Remove all build directories before building"
    Write-Host "  -RunSampleTests         Run sample tests after building them"
    Write-Host "  -NoTests                Library-only build: skip samples, unit tests and mlss-tester"
    Write-Host "  -NoSampleTests          Skip building the samples"
    Write-Host "  -NoUnitTests            Skip building the unit tests (and mlss-tester)"
    Write-Host ""
    Write-Host "  By default every build also builds the samples and unit tests"
    Write-Host "  (which pulls in mlss-tester). Use -NoTests for a library-only build,"
    Write-Host "  or -NoSampleTests / -NoUnitTests to skip just one."
    Write-Host "  -d, -Deploy             Deploy amdmlss and cmake files after building"
    Write-Host "  -DeployPath <path>      Override the deployment directory (default: ./amdmlss_redist)"
    Write-Host "  -Help, -h               Show this help message"
    Write-Host ""
    Write-Host "  When using -c all, builds with all supported compilers."
    Write-Host "  When using -b all, builds both debug and release configurations."
    Write-Host "  Each build will be placed in build/<compiler>-<build_type> directory."
    exit 0
}

# HIP_PATH / ROCM_PATH must be supplied by the environment so the build
# adapts to whatever ROCm install the user has. At least one of them must
# be set; if only one is provided we mirror it to the other so downstream
# CMake projects can pick whichever name they prefer.
if (-not $env:HIP_PATH -and -not $env:ROCM_PATH) {
    Write-Host "ERROR: neither HIP_PATH nor ROCM_PATH is set in the environment." -ForegroundColor Red
    Write-Host "       Set one of them to your ROCm install root, for example:"
    Write-Host "           `$env:HIP_PATH  = 'C:/opt/rocm'"
    Write-Host "           `$env:ROCM_PATH = 'C:/opt/rocm'"
    Write-Host "       Or persist it across sessions:"
    Write-Host "           setx HIP_PATH C:/opt/rocm"
    exit 1
}
if (-not $env:HIP_PATH)  { $env:HIP_PATH  = $env:ROCM_PATH }
if (-not $env:ROCM_PATH) { $env:ROCM_PATH = $env:HIP_PATH }

# The test dependencies (Catch2, amd-mlss-tester) are no longer git submodules;
# CMake fetches them with FetchContent, and only when the unit tests are built.

# Handle clean-up if requested
if ($CleanUp) {
    Write-Host "Cleaning up build directories..."
    
    $dirsToRemove = @("build", "build-clang", "build_clang", "build-vs", "build_vs", "buildvsdebug")
    
    foreach ($dir in $dirsToRemove) {
        if (Test-Path $dir) {
            Remove-Item -Path $dir -Recurse -Force
            Write-Host "Removed $dir directory"
        }
    }
    
    Write-Host "Clean-up completed!"
    Write-Host ""
}

# Map a preset name to its build configuration (Debug/Release). The name may
# carry a test-mode suffix (-none/-unit/-samples), so match the build type
# substring rather than relying on token position.
function Get-BuildConfig {
    param([string]$Preset)
    if ($Preset -like '*debug*') { return 'Debug' }
    return 'Release'
}

# Select the preset variant matching the requested unit/sample combination.
# The base preset builds both; the suffixed variants restrict the matrix.
function Get-PresetVariant {
    param([string]$BasePreset)
    $unit    = -not ($NoTests -or $NoUnitTests)
    $samples = -not ($NoTests -or $NoSampleTests)
    if ($unit -and $samples) { return $BasePreset }
    if ($unit)               { return "$BasePreset-unit" }
    if ($samples)            { return "$BasePreset-samples" }
    return "$BasePreset-none"
}

# Function to deploy amdmlss
function Deploy-Amdmlss {
    param(
        [string]$Preset,
        [string]$DeployPath
    )
    
    $buildConfigCapitalized = Get-BuildConfig $Preset
    
    Write-Host "Deploying amdmlss to: $DeployPath"
    
    # Create deployment directory
    if (-not (Test-Path $DeployPath)) {
        New-Item -ItemType Directory -Path $DeployPath -Force | Out-Null
    }
    
    # Install using CMake with the correct configuration
    cmake --install "build/$Preset" --config $buildConfigCapitalized --prefix $DeployPath | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Deployment failed for $Preset!"
        return $false
    }
    
    Write-Host "Deployment completed successfully!"
    return $true
}

# Function to build with a single preset (base preset, e.g. clang-release;
# the matching test-mode variant is selected from the -No* switches).
function Build-Single {
    param(
        [string]$Preset
    )

    $buildConfigCapitalized = Get-BuildConfig $Preset

    # Samples and unit tests are built on every build by default. -NoTests
    # opts out of both; -NoSampleTests / -NoUnitTests opt out of each one.
    # Only the unit tests pull in Catch2 + mlss-tester (fetched by CMake).
    $buildSamples = -not ($NoTests -or $NoSampleTests)
    $buildUnitTests = -not ($NoTests -or $NoUnitTests)

    # The test/sample combination is expressed entirely through the preset
    # variant; CMake fetches the test dependencies only for unit-test builds.
    $presetVariant = Get-PresetVariant $Preset
    $script:LastBuiltPreset = $presetVariant

    Write-Host ""
    Write-Host "Configuring with CMake preset: $presetVariant..."
    cmake --preset $presetVariant | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed for $presetVariant!"
        return $false
    }
    
    # Build the project (including unit tests when enabled)
    Write-Host "Building project..."
    cmake --build "build/$presetVariant" --config $buildConfigCapitalized | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed for $presetVariant!"
        return $false
    }
    
    Write-Host "$presetVariant build completed successfully!"

    if ($buildSamples) {
        Write-Host "Samples built successfully!"
    }
    if ($buildUnitTests) {
        Write-Host "Unit tests built successfully!"
    }

    # Run sample tests if requested. The samples are already built as part of
    # the main build above (unless the samples were disabled), so this only
    # executes them.
    if ($RunSampleTests -and (-not $buildSamples)) {
        Write-Host "-RunSampleTests ignored because the samples were not built."
    }
    elseif ($RunSampleTests) {
        Write-Host ""
        Write-Host "Running sample tests for $presetVariant..."
        
        # First try the base samples directory (for Ninja builds)
        $testDir = "build/$presetVariant/samples"
        
        # Check if exe files exist in base directory
        $hasExeFiles = $false
        if (Test-Path $testDir) {
            $exeFiles = Get-ChildItem -Path $testDir -Filter "*.exe" -ErrorAction SilentlyContinue
            if ($exeFiles) {
                $hasExeFiles = $true
            }
        }
        
        # If not found, try with build configuration subdirectory (for MSBuild)
        if (-not $hasExeFiles) {
            $testDir = "build/$presetVariant/samples/$buildConfigCapitalized"
            if (-not (Test-Path $testDir)) {
                $testDir = "build/$presetVariant/samples/Debug"
            }
            if (-not (Test-Path $testDir)) {
                $testDir = "build/$presetVariant/samples/Release"
            }
        }
        
        if (Test-Path $testDir) {
            $testFailed = $false
            $testExes = Get-ChildItem -Path $testDir -Filter "*.exe" -ErrorAction SilentlyContinue
            
            foreach ($testExe in $testExes) {
                Write-Host "Running test: $($testExe.BaseName)"
                & $testExe.FullName
                
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Test $($testExe.BaseName) failed!"
                    $testFailed = $true
                } else {
                    Write-Host "Test $($testExe.BaseName) passed!"
                }
                Write-Host ""
            }
            
            if ($testFailed) {
                Write-Host "Some tests failed for $presetVariant!"
                return $false
            }
            Write-Host "All sample tests passed for $presetVariant!"
        } else {
            Write-Host "No test directory found at $testDir"
            Write-Host "Skipping sample tests for $presetVariant"
        }
    }
    
    return $true
}

# Determine which build types to build
if ($Build -eq 'all') {
    $buildTypes = @('debug', 'release')
    Write-Host "Building both debug and release configurations..."
} else {
    $buildTypes = @($Build)
}

# Compilers to iterate over for "-c all" — must mirror the '.bat' / '.sh' lists.
$supportedCompilers = @('vs2026', 'vs2022', 'clang')

# Handle "all" compiler option
if ($Compiler -eq 'all') {
    Write-Host "Building with all supported compilers..."
    Write-Host ""

    foreach ($buildType in $buildTypes) {
        foreach ($compiler in $supportedCompilers) {
            $preset = "$compiler-$buildType"
            Write-Host "========================================"
            Write-Host "Building with $preset"
            Write-Host "========================================"
            $result = Build-Single -Preset $preset
            if (-not $result) {
                Write-Host "$preset build failed!"
                exit 1
            }
            Write-Host ""
        }
    }

    Write-Host "All builds completed successfully!"

    # Deploy if requested (use the last built configuration)
    if ($Deploy) {
        Write-Host ""
        Write-Host "========================================"
        Write-Host "Deploying amdmlss"
        Write-Host "========================================"
        $result = Deploy-Amdmlss -Preset $script:LastBuiltPreset -DeployPath $DeployPath
        if (-not $result) {
            Write-Host "Deployment failed!"
            exit 1
        }
    }

    exit 0
}

# Build with single compiler (possibly multiple build types)
if ($buildTypes.Count -gt 1) {
    Write-Host "Building with compiler: $Compiler"
    Write-Host ""
}

$lastPreset = ""
foreach ($buildType in $buildTypes) {
    $preset = "$Compiler-$buildType"
    
    if ($buildTypes.Count -gt 1) {
        Write-Host "========================================"
        Write-Host "Building with preset: $preset"
        Write-Host "========================================"
    } else {
        Write-Host "Building with preset: $preset"
    }
    
    $result = Build-Single -Preset $preset
    if (-not $result) {
        Write-Host "Build failed!"
        exit 1
    }
    
    if ($buildTypes.Count -gt 1) {
        Write-Host ""
    }
}

Write-Host "Build completed successfully!"

# Deploy if requested
if ($Deploy) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Deploying amdmlss"
    Write-Host "========================================"
    $result = Deploy-Amdmlss -Preset $script:LastBuiltPreset -DeployPath $DeployPath
    if (-not $result) {
        Write-Host "Deployment failed!"
        exit 1
    }
}

# If no parameters provided, suggest using -c all
if ($PSBoundParameters.Count -eq 0) {
    Write-Host ""
    Write-Host "Tip: Use '.\$($MyInvocation.MyCommand.Name) -c all' to build with all supported compilers"
}
