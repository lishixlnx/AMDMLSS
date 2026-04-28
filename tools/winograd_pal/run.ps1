<#
.SYNOPSIS
    End-to-end orchestrator: extract -> link -> reemit.

.DESCRIPTION
    Runs the three Winograd PAL pipeline steps in sequence:

      1. python tools/winograd_pal/extract.py
      2. tools/winograd_pal/link.ps1
      3. python tools/winograd_pal/reemit.py

    Cleans `<repo>/build/winograd_pal` first (idempotent re-runs). Any step
    returning a non-zero exit code stops the pipeline.

.PARAMETER BuildRoot
    Output directory (default: <repo>/build/winograd_pal).

.PARAMETER Lld
    Optional explicit path to `ld.lld`.

.PARAMETER Readelf
    Optional explicit path to `llvm-readelf` (or `llvm-readobj`).

.PARAMETER OnlyKernel
    If set, links only the named kernel. Useful for the smoke-test gate;
    skips the reemit step (since per-directory headers would be incomplete).
#>

[CmdletBinding()]
param(
    [string] $BuildRoot,
    [string] $Lld,
    [string] $Readelf,
    [string] $OnlyKernel
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
if (-not $BuildRoot) {
    $BuildRoot = Join-Path $repoRoot "build/winograd_pal"
}

Write-Host "==> winograd_pal pipeline"
Write-Host "    repo root: $repoRoot"
Write-Host "    build:     $BuildRoot"
Write-Host ""

Write-Host "==> [1/3] extract"
$extractArgs = @("--out", $BuildRoot)
& python (Join-Path $scriptDir "extract.py") @extractArgs
if ($LASTEXITCODE -ne 0) { throw "extract.py failed (exit $LASTEXITCODE)" }
Write-Host ""

Write-Host "==> [2/3] link"
$linkArgs = @("-BuildRoot", $BuildRoot)
if ($Lld)        { $linkArgs += @("-Lld", $Lld) }
if ($Readelf)    { $linkArgs += @("-Readelf", $Readelf) }
if ($OnlyKernel) { $linkArgs += @("-OnlyKernel", $OnlyKernel) }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scriptDir "link.ps1") @linkArgs
if ($LASTEXITCODE -ne 0) { throw "link.ps1 failed (exit $LASTEXITCODE)" }
Write-Host ""

if ($OnlyKernel) {
    Write-Host "==> [3/3] reemit SKIPPED (--OnlyKernel set; per-directory output would be incomplete)"
    return
}

Write-Host "==> [3/3] reemit"
& python (Join-Path $scriptDir "reemit.py") "--build-root" $BuildRoot
if ($LASTEXITCODE -ne 0) { throw "reemit.py failed (exit $LASTEXITCODE)" }
Write-Host ""
Write-Host "==> done"
