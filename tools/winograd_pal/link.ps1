<#
.SYNOPSIS
    Drive ld.lld to link every Winograd PAL relocatable extracted by
    extract.py into a shared (ET_DYN) AMDPAL ELF.

.DESCRIPTION
    Walks every manifest.json under -BuildRoot, runs:

        ld.lld --shared -m elf64_amdgpu --no-undefined --gc-sections \
               -Bsymbolic --pie <NAME>.o -o <NAME>.linked.elf

    captures stdout/stderr into <NAME>.link.log, then asserts via
    llvm-readelf that the result is ET_DYN, OS/ABI=AMDGPU - AMD PAL,
    Machine=EM_AMDGPU, with the expected lower-byte mach in e_flags.

    Any failure stops the script with a non-zero exit code.

.PARAMETER BuildRoot
    Directory containing per-arch subdirectories with `manifest.json` files
    (default: <repo>/build/winograd_pal).

.PARAMETER Lld
    Path to `ld.lld`. Defaults to lookup on PATH.

.PARAMETER Readelf
    Path to `llvm-readelf`. Defaults to lookup on PATH.

.PARAMETER OnlyKernel
    If set, only the kernel with this exact symbol name is linked. Useful
    for the smoke-test gate.
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

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $BuildRoot) {
    $BuildRoot = Join-Path $repoRoot "build/winograd_pal"
}

function Resolve-Tool {
    param(
        [string] $Override,
        [string[]] $Candidates,
        [string] $FriendlyName
    )
    if ($Override) {
        if (-not (Test-Path $Override)) {
            throw "$FriendlyName not found at: $Override"
        }
        return (Resolve-Path $Override).Path
    }
    foreach ($cand in $Candidates) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($null -ne $cmd) {
            return $cmd.Path
        }
    }
    $candList = ($Candidates -join ", ")
    throw @"
$FriendlyName not found on PATH (looked for: $candList).
Install ROCm-for-Windows or build LLVM with LLVM_TARGETS_TO_BUILD=AMDGPU,
then either prepend its bin/ to PATH or pass an explicit path.
"@
}

function Invoke-ReadElfHeader {
    param([string] $ToolPath, [string] $ElfPath)
    # Both `llvm-readelf` and `llvm-readobj` accept `--elf-output-style=GNU`
    # and produce the GNU-style ELF header dump expected by Assert-LinkedElf.
    return & $ToolPath --elf-output-style=GNU -h $ElfPath 2>&1
}

# ELFOSABI_AMDGPU_PAL = 0x41; lives at e_ident[7].
$EI_OSABI = 7
$ELFOSABI_AMDGPU_PAL = 0x41

function Set-PalOsAbi {
    param([string] $ElfPath, [string] $KernelName)
    $bytes = [System.IO.File]::ReadAllBytes($ElfPath)
    if ($bytes.Length -lt 16 -or $bytes[0] -ne 0x7f -or $bytes[1] -ne 0x45 -or $bytes[2] -ne 0x4c -or $bytes[3] -ne 0x46) {
        throw "$KernelName`: linked file is not an ELF"
    }
    if ($bytes[$EI_OSABI] -ne $ELFOSABI_AMDGPU_PAL) {
        $bytes[$EI_OSABI] = $ELFOSABI_AMDGPU_PAL
        [System.IO.File]::WriteAllBytes($ElfPath, $bytes)
    }
}

function Get-AmdGpuMachByte {
    param([string] $ElfPath)
    # ELF64 e_flags lives at offset 0x30; AMDGPU mach code is the low byte.
    $bytes = [System.IO.File]::ReadAllBytes($ElfPath)
    if ($bytes.Length -lt 0x34) {
        throw "$ElfPath`: ELF too small to read e_flags"
    }
    return $bytes[0x30]
}

function Assert-LinkedElf {
    param(
        [string] $ReadelfPath,
        [string] $ElfPath,
        [string] $SourceObjPath,
        [string] $KernelName
    )

    $readelfOut = Invoke-ReadElfHeader -ToolPath $ReadelfPath -ElfPath $ElfPath
    if ($LASTEXITCODE -ne 0) {
        throw "readelf -h failed for $KernelName`:`n$readelfOut"
    }
    $joined = ($readelfOut -join "`n")

    if ($joined -notmatch "Type:\s+DYN") {
        throw "$KernelName`: expected Type: DYN (Shared object), got:`n$joined"
    }
    if ($joined -notmatch "OS/ABI:\s+AMDGPU\s*-\s*PAL") {
        throw "$KernelName`: expected OS/ABI: AMDGPU - PAL, got:`n$joined"
    }
    if ($joined -notmatch "EM_AMDGPU") {
        throw "$KernelName`: expected Machine: EM_AMDGPU, got:`n$joined"
    }

    $flagsMatch = [regex]::Match($joined, "Flags:\s+0x([0-9a-fA-F]+)")
    if (-not $flagsMatch.Success) {
        throw "$KernelName`: could not parse Flags from llvm-readelf output"
    }
    $linkedMach = ([Convert]::ToUInt32($flagsMatch.Groups[1].Value, 16)) -band 0xFF
    $sourceMach = Get-AmdGpuMachByte -ElfPath $SourceObjPath
    if ($linkedMach -ne $sourceMach) {
        throw ("$KernelName`: e_flags mach changed during link (source=0x{0:X2}, linked=0x{1:X2})" -f $sourceMach, $linkedMach)
    }
}

function Invoke-Link {
    param(
        [string] $LldPath,
        [string] $ReadelfPath,
        [string] $ManifestPath,
        [string] $OnlyKernel
    )

    $manifest = Get-Content -Raw $ManifestPath | ConvertFrom-Json
    $dir = Split-Path -Parent $ManifestPath
    $linked = 0
    foreach ($k in $manifest.kernels) {
        if ($OnlyKernel -and ($k.name -ne $OnlyKernel)) { continue }

        $obj = Join-Path $dir ([System.IO.Path]::GetFileName($k.dst_o))
        $out = [System.IO.Path]::ChangeExtension($obj, ".linked.elf")
        $log = [System.IO.Path]::ChangeExtension($obj, ".link.log")

        if (-not (Test-Path $obj)) {
            throw "missing extracted object: $obj"
        }

        # Notes on the flag set:
        #   --shared          emit ET_DYN (the whole point of this tool).
        #   --no-undefined    treat unresolved references as a hard error;
        #                      the PAL pipelines are entirely self-contained.
        #   -Bsymbolic        bind references to local definitions, mirroring
        #                      what comgr LINK_RELOCATABLE_TO_EXECUTABLE does
        #                      for HSA inputs.
        # `--gc-sections` is intentionally NOT used: the kernel entry point
        # `_amdgpu_cs_main` is a LOCAL symbol, so GC would treat its .text
        # section as unreachable and emit an empty stub.
        # `--pie` is intentionally NOT used: ROCm's ld.lld treats it as
        # incompatible with `--shared`.
        $lldArgs = @(
            "--shared",
            "-m", "elf64_amdgpu",
            "--no-undefined",
            "-Bsymbolic",
            $obj,
            "-o", $out
        )

        Write-Host "  linking $($manifest.family)/$($manifest.arch)/$($manifest.precision)/$($k.name)"
        & $LldPath @lldArgs 2>&1 | Tee-Object -FilePath $log | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "ld.lld failed for $($k.name); see $log"
        }

        # Restore the OS/ABI byte to AMDGPU_PAL (0x41). LLD always emits
        # 0x40 (HSA) for `--shared` outputs regardless of input OSABI; the
        # PAL pipeline metadata is preserved verbatim in the .note section,
        # so flipping this byte is sufficient to let downstream PAL
        # consumers recognize the file as a PAL pipeline.
        Set-PalOsAbi -ElfPath $out -KernelName $k.name

        Assert-LinkedElf -ReadelfPath $ReadelfPath -ElfPath $out -SourceObjPath $obj -KernelName $k.name
        $linked++
    }
    return $linked
}

$lldPath = Resolve-Tool -Override $Lld -Candidates @("ld.lld", "ld.lld.exe") -FriendlyName "ld.lld"
$readelfPath = Resolve-Tool -Override $Readelf -Candidates @("llvm-readelf", "llvm-readelf.exe", "llvm-readobj", "llvm-readobj.exe") -FriendlyName "llvm-readelf/llvm-readobj"

Write-Host "ld.lld:       $lldPath"
Write-Host "llvm-readelf: $readelfPath"
Write-Host "build root:   $BuildRoot"

if (-not (Test-Path $BuildRoot)) {
    throw "build root does not exist: $BuildRoot. Run extract.py first."
}

$manifests = Get-ChildItem -Path $BuildRoot -Recurse -Filter "manifest.json"
if ($manifests.Count -eq 0) {
    throw "no manifest.json found under $BuildRoot. Run extract.py first."
}

$total = 0
foreach ($m in $manifests) {
    $total += (Invoke-Link -LldPath $lldPath -ReadelfPath $readelfPath -ManifestPath $m.FullName -OnlyKernel $OnlyKernel)
}

Write-Host ""
if ($OnlyKernel) {
    Write-Host "Linked $total kernel(s) matching name '$OnlyKernel'."
} else {
    Write-Host "Linked $total kernel(s) successfully."
}
