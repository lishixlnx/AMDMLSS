[CmdletBinding()]
param(
    [Parameter()]
    [Alias('c')]
    [ValidateSet('vs2022', 'clang', 'all')]
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
    [switch]$BuildSampleTests,
    
    [Parameter()]
    [switch]$RunSampleTests,
    
    [Parameter()]
    [switch]$BuildAllTests,
    
    [Parameter()]
    [Alias('d')]
    [string]$Deploy = '',
    
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
    Write-Host "  -c, -Compiler           Compiler: vs2022, clang, all (default: clang)"
    Write-Host "  -b, -Build              Build type: debug, release, all (default: release)"
    Write-Host "  -CleanUp                Remove all build directories before building"
    Write-Host "  -BuildSampleTests       Build sample tests but don't run them"
    Write-Host "  -RunSampleTests         Build and run sample tests (or just run if already built)"
    Write-Host "  -BuildAllTests          Build sample tests, mlss-tester and unit tests"
    Write-Host "  -d, -Deploy [path]      Deploy amdmlss and cmake files (default: ./amdmlss_redist)"
    Write-Host "  -Help, -h               Show this help message"
    Write-Host ""
    Write-Host "  When using -c all, builds with all supported compilers."
    Write-Host "  When using -b all, builds both debug and release configurations."
    Write-Host "  Each build will be placed in build/<compiler>-<build_type> directory."
    exit 0
}

# Set default deploy path if deploy is requested without path
if ($Deploy -eq 'true' -or $Deploy -eq '1') {
    $Deploy = './amdmlss_redist'
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
    
    # Configure with CMake preset
    Write-Host "Configuring with CMake preset: $Preset..."
    cmake --preset $Preset | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed for $Preset!"
        return $false
    }
    
    # Build the project
    Write-Host "Building project..."
    cmake --build "build/$Preset" --config $buildConfigCapitalized | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed for $Preset!"
        return $false
    }
    
    Write-Host "$Preset build completed successfully!"

    # Build all tests (mlss-tester + unit tests + sample tests)
    if ($BuildAllTests) {
        Write-Host ""
        Write-Host "Building amd-mlss-tester library..."

        $testerSrc = "3rdparty/amd-mlss-tester"
        $testerBuild = "build/mlss-tester-$Preset"
        $testerInstall = "$testerSrc/install"

        $testerArgs = @(
            "-DCMAKE_BUILD_TYPE=$buildConfigCapitalized",
            "-DCMAKE_INSTALL_PREFIX=$testerInstall",
            "-DMLSS_ENABLE_HIP=ON",
            "-DBUILD_APP=OFF",
            "-DBUILD_TESTING=OFF"
        )

        if ($Preset -like 'clang-*') {
            $hipPath = if ($env:HIP_PATH) { $env:HIP_PATH } else { 'C:/opt/rocm' }
            $testerArgs += @(
                "-G", "Ninja",
                "-DCMAKE_CXX_COMPILER=$hipPath/bin/clang++.exe",
                "-DCMAKE_CXX_FLAGS=-Wno-unused-command-line-argument"
            )
        } else {
            $testerArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
        }

        & cmake @testerArgs -S $testerSrc -B $testerBuild | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "amd-mlss-tester configuration failed!"
            return $false
        }

        cmake --build $testerBuild --config $buildConfigCapitalized | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "amd-mlss-tester build failed!"
            return $false
        }

        cmake --install $testerBuild --config $buildConfigCapitalized | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "amd-mlss-tester install failed!"
            return $false
        }
        Write-Host "amd-mlss-tester built and installed successfully!"

        Write-Host ""
        Write-Host "Reconfiguring with BUILD_SAMPLES=ON for $Preset..."
        cmake --preset $Preset -DBUILD_SAMPLES=ON | Out-Host

        if ($LASTEXITCODE -ne 0) {
            Write-Host "CMake reconfiguration failed for $Preset!"
            return $false
        }

        Write-Host "Building all tests for $Preset..."
        cmake --build "build/$Preset" --config $buildConfigCapitalized | Out-Host

        if ($LASTEXITCODE -ne 0) {
            Write-Host "All-tests build failed for $Preset!"
            return $false
        }
        Write-Host "All tests built successfully!"
    }

    # Build sample tests if requested
    if ($BuildSampleTests -or $RunSampleTests) {
        Write-Host ""
        Write-Host "Reconfiguring with BUILD_SAMPLES=ON for $Preset..."
        cmake --preset $Preset -DBUILD_SAMPLES=ON | Out-Host
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CMake reconfiguration failed for $Preset!"
            return $false
        }
        
        Write-Host "Building sample tests for $Preset..."
        cmake --build "build/$Preset" --config $buildConfigCapitalized | Out-Host
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Sample tests build failed for $Preset!"
            return $false
        }
        Write-Host "Sample tests built successfully!"
    }
    
    # Run sample tests if requested
    if ($RunSampleTests) {
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

# Handle "all" compiler option
if ($Compiler -eq 'all') {
    Write-Host "Building with all supported compilers..."
    Write-Host ""
    
    # Build all combinations
    foreach ($buildType in $buildTypes) {
        # Build with vs2022
        Write-Host "========================================"
        Write-Host "Building with vs2022-$buildType"
        Write-Host "========================================"
        $result = Build-Single -Preset "vs2022-$buildType"
        if (-not $result) {
            Write-Host "vs2022-$buildType build failed!"
            exit 1
        }
        Write-Host ""
        
        # Build with clang
        Write-Host "========================================"
        Write-Host "Building with clang-$buildType"
        Write-Host "========================================"
        $result = Build-Single -Preset "clang-$buildType"
        if (-not $result) {
            Write-Host "clang-$buildType build failed!"
            exit 1
        }
        Write-Host ""
    }
    
    Write-Host "All builds completed successfully!"
    
    # Deploy if requested (use the last built configuration)
    if ($Deploy) {
        Write-Host ""
        Write-Host "========================================"
        Write-Host "Deploying amdmlss"
        Write-Host "========================================"
        $lastPreset = "clang-$($buildTypes[-1])"
        $result = Deploy-Amdmlss -Preset $lastPreset -DeployPath $Deploy
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
    $result = Deploy-Amdmlss -Preset $lastPreset -DeployPath $Deploy
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
