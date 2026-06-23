# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
param(
    [string[]] $Presets = @(
        'vs2022-debug',
        'vs2022-release',
        'vs2026-debug',
        'vs2026-release',
        'clang-debug',
        'clang-release'
    )
)

$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $repoRoot

$logRoot = Join-Path $repoRoot 'build\_logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Get-Cfg([string]$preset) {
    if ($preset -like '*-debug') { 'Debug' } else { 'Release' }
}

# Tee-Object would normally also forward the captured objects down the
# pipeline, which contaminates a function's return value. Redirect the tee'd
# output to Out-Host so only the integer exit code crosses the function
# boundary.
function Invoke-CmakeStep {
    param(
        [string]   $label,
        [string]   $logFile,
        [string[]] $cmakeArgs
    )
    Write-Host ""
    Write-Host "===================================================================="
    Write-Host "$label"
    Write-Host "===================================================================="
    Write-Host "$ cmake $($cmakeArgs -join ' ')"
    & cmake @cmakeArgs 2>&1 | Tee-Object -FilePath $logFile | Out-Host
    return $LASTEXITCODE
}

function Invoke-CTestStep {
    param(
        [string]   $label,
        [string]   $logFile,
        [string[]] $ctestArgs
    )
    Write-Host ""
    Write-Host "===================================================================="
    Write-Host "$label"
    Write-Host "===================================================================="
    Write-Host "$ ctest $($ctestArgs -join ' ')"
    & ctest @ctestArgs 2>&1 | Tee-Object -FilePath $logFile | Out-Host
    return $LASTEXITCODE
}

$summary = @()

foreach ($preset in $Presets) {
    $cfg = Get-Cfg $preset
    $configLog = Join-Path $logRoot "$preset.configure.log"
    $buildLog  = Join-Path $logRoot "$preset.build.log"
    $testLog   = Join-Path $logRoot "$preset.test.log"
    $buildDir  = Join-Path $repoRoot "build\$preset"

    # CMAKE_SUPPRESS_REGENERATION=ON avoids the "Cannot restore timestamp ...
    # generate.stamp" race that hits the Visual Studio generator when many
    # parallel targets fire CMake re-runs simultaneously. Safe here because
    # this script always reconfigures before building.
    $configRc = Invoke-CmakeStep `
        "[$preset] CONFIGURE (-DBUILD_SAMPLE_TESTS=ON -DBUILD_UNIT_TESTS=ON)" `
        $configLog `
        @('--preset', $preset, '-DBUILD_SAMPLE_TESTS=ON', '-DBUILD_UNIT_TESTS=ON',
          '-DCMAKE_SUPPRESS_REGENERATION=ON')

    $buildRc = -1
    $testRc  = -1

    if ($configRc -eq 0) {
        $buildRc = Invoke-CmakeStep `
            "[$preset] BUILD ($cfg)" `
            $buildLog `
            @('--build', $buildDir, '--config', $cfg, '--parallel')

        if ($buildRc -eq 0) {
            $testRc = Invoke-CTestStep `
                "[$preset] TEST ($cfg)" `
                $testLog `
                @('--test-dir', $buildDir, '-C', $cfg, '--output-on-failure')
        } else {
            Write-Host "[$preset] BUILD FAILED (rc=$buildRc) - skipping tests"
        }
    } else {
        Write-Host "[$preset] CONFIGURE FAILED (rc=$configRc) - skipping build/tests"
    }

    $summary += [PSCustomObject]@{
        Preset    = $preset
        Configure = $configRc
        Build     = $buildRc
        Test      = $testRc
    }
}

Write-Host ""
Write-Host "===================================================================="
Write-Host "FINAL SUMMARY"
Write-Host "===================================================================="
$summary | Format-Table -AutoSize | Out-String | Write-Host
$summary | Export-Csv -Path (Join-Path $logRoot 'summary.csv') -NoTypeInformation -Force
