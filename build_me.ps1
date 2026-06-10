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

# Update submodules to the latest tracked branch tip. The build requires the
# submodules to be at their latest tracked tip, so a failure here is fatal.
#
# git writes its progress and errors to stderr; redirect it into the success
# stream so PowerShell does not promote it to a terminating NativeCommandError
# (under $ErrorActionPreference='Stop') before we can report a clear message.
Write-Host "Updating submodules..."
try {
    git submodule update --remote 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: git submodule update --remote failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "ERROR: git submodule update --remote failed ($($_.Exception.Message))!" -ForegroundColor Red
    exit 1
}

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

# Initialise submodules to the SHAs pinned by this checkout. We deliberately
# do NOT run "git submodule update --remote": pulling the tip of each tracked
# remote branch mutates the working tree, requires network access, and makes
# builds non-reproducible (and fails in offline/CI environments). Builds use
# the pinned SHAs; bump them explicitly with a separate, intentional commit.
Write-Host "Initialising git submodules..."
git submodule update --init --recursive | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "git submodule update --recursive failed!"
    exit 1
}
Write-Host ""

# Function to deploy amdmlss
function Deploy-Amdmlss {
    param(
        [string]$Preset,
        [string]$DeployPath
    )
    
    # Extract build type from preset name
    $buildConfig = $Preset.Split('-')[-1]
    $buildConfigCapitalized = (Get-Culture).TextInfo.ToTitleCase($buildConfig)
    
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

# Function to build with a single preset
function Build-Single {
    param(
        [string]$Preset
    )
    
    # Extract build type from preset name
    $buildConfig = $Preset.Split('-')[-1]
    $buildConfigCapitalized = (Get-Culture).TextInfo.ToTitleCase($buildConfig)

    # Samples and unit tests are built on every build by default. -NoTests
    # opts out of both; -NoSampleTests / -NoUnitTests opt out of each one.
    # Only the unit tests link mlss-tester, so the tester is built only when
    # the unit tests are enabled.
    $buildSamples = -not ($NoTests -or $NoSampleTests)
    $buildUnitTests = -not ($NoTests -or $NoUnitTests)

    # Build mlss-tester BEFORE the main project so the correct config is installed
    if ($buildUnitTests) {
        Write-Host ""
        Write-Host "Building amd-mlss-tester library..."

        $testerSrc = "3rdparty/amd-mlss-tester"
        $testerInstall = "$testerSrc/install"

        # Map main-project preset to the matching mlss-tester preset
        $testerPreset = switch -Wildcard ($Preset) {
            'clang-debug'      { 'clang-lib-static-debug'   }
            'clang-release'    { 'clang-lib-static-release'  }
            'vs2022-*'         { 'vs2022-lib-static'         }
            'vs2026-*'         { 'vs2026-lib-static'         }
            default            { '' }
        }

        if ($testerPreset -eq '') {
            Write-Host "No matching mlss-tester preset for '$Preset'!"
            return $false
        }

        if (-not $env:HIP_PATH) { $env:HIP_PATH = 'C:/opt/rocm' }

        $dx12IncludeDir = Join-Path $PSScriptRoot '3rdparty/amd-mlss-tester/3rdparty/amd-cross-compiler-tester/lib/include'
        $dx12SourceDir  = Join-Path $PSScriptRoot '3rdparty/amd-mlss-tester/3rdparty/amd-cross-compiler-tester/lib/src'

        $testerConfigArgs = @(
            "--preset", $testerPreset,
            "-DCMAKE_INSTALL_PREFIX=$testerInstall",
            "-DMLSS_ENABLE_HIP=ON",
            "-DMLSS_ENABLE_AOCL=ON",
            "-DMLSS_DX12_INCLUDE_DIR=$dx12IncludeDir",
            "-DMLSS_DX12_SOURCE_DIR=$dx12SourceDir",
            "-DBUILD_APP=OFF",
            "-DBUILD_TESTING=OFF"
        )

        if ($Preset -like 'clang-*') {
            $testerConfigArgs += @(
                "-DCMAKE_CXX_COMPILER=$env:HIP_PATH/bin/clang++.exe",
                "-DCMAKE_CXX_FLAGS=-Wno-unused-command-line-argument"
            )
        }

        cmake @testerConfigArgs -S $testerSrc | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "amd-mlss-tester configuration failed!"
            return $false
        }

        cmake --build "$testerSrc/build/$testerPreset" --config $buildConfigCapitalized | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "amd-mlss-tester build failed!"
            return $false
        }

        cmake --install "$testerSrc/build/$testerPreset" --config $buildConfigCapitalized | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "amd-mlss-tester install failed!"
            return $false
        }
        Write-Host "amd-mlss-tester built and installed successfully!"
    }

    # Configure with CMake preset.
    #
    # BUILD_TESTS / BUILD_SAMPLES are driven explicitly by this script rather
    # than relying on the CMake defaults. Both default to ON so every build
    # produces the samples and unit tests (the latter linking mlss-tester,
    # built and installed above). The opt-out flags turn each OFF; disabling
    # the unit tests also skips the mlss-tester/AOCL dependency.
    Write-Host ""
    Write-Host "Configuring with CMake preset: $Preset..."
    $configArgs = @("--preset", $Preset)
    if ($buildUnitTests) {
        $configArgs += "-DBUILD_TESTS=ON"
    } else {
        $configArgs += "-DBUILD_TESTS=OFF"
    }
    if ($buildSamples) {
        $configArgs += "-DBUILD_SAMPLES=ON"
    } else {
        $configArgs += "-DBUILD_SAMPLES=OFF"
    }
    cmake @configArgs | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed for $Preset!"
        return $false
    }
    
    # Build the project (including unit tests when mlss-tester is available)
    Write-Host "Building project..."
    cmake --build "build/$Preset" --config $buildConfigCapitalized | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed for $Preset!"
        return $false
    }
    
    Write-Host "$Preset build completed successfully!"

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
        Write-Host "Running sample tests for $Preset..."
        
        # First try the base samples directory (for Ninja builds)
        $testDir = "build/$Preset/samples"
        
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
            $testDir = "build/$Preset/samples/$buildConfigCapitalized"
            if (-not (Test-Path $testDir)) {
                $testDir = "build/$Preset/samples/Debug"
            }
            if (-not (Test-Path $testDir)) {
                $testDir = "build/$Preset/samples/Release"
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
                Write-Host "Some tests failed for $Preset!"
                return $false
            }
            Write-Host "All sample tests passed for $Preset!"
        } else {
            Write-Host "No test directory found at $testDir"
            Write-Host "Skipping sample tests for $Preset"
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
        $lastPreset = "$($supportedCompilers[-1])-$($buildTypes[-1])"
        $result = Deploy-Amdmlss -Preset $lastPreset -DeployPath $DeployPath
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
    
    $lastPreset = $preset
    
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
    $result = Deploy-Amdmlss -Preset $lastPreset -DeployPath $DeployPath
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
