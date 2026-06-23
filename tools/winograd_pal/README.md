<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. -->

# `tools/winograd_pal/`

Offline tooling that converts the relocatable AMDPAL ELFs embedded in every
`shadersBinReloc.hpp` under
`modules/shaders/src/operators/impl/conv/mxn/Winograd/` into linked
(`ET_DYN`, `OS/ABI=AMDGPU_PAL`) PAL pipelines, emitted into sibling
`shadersBinNonReloc.hpp` files.

Why this exists: the runtime path that produces "non-relocatable" Winograd
binaries goes through `comgr`'s `LINK_RELOCATABLE_TO_EXECUTABLE` action,
which is HSA-only. The Winograd ELFs are PAL pipelines (`OS/ABI=0x41`),
so `comgr` exits silently on them, taking the host process with it. The
fix is to produce linked PAL `ET_DYN` outputs offline with `ld.lld` and
ship them alongside the existing relocatables.

## What gets produced

For every input header below, a sibling `shadersBinNonReloc.hpp` is written
in the same directory with one `_NonReloc`-suffixed `StaticShaderType`
entry per kernel:

| Source                                                                                      | Kernels |
| ------------------------------------------------------------------------------------------- | ------- |
| `Winograd/Base/gfx1100/fp16/shadersBinReloc.hpp`                                            | 6       |
| `Winograd/Base/gfx1100/fp32/shadersBinReloc.hpp`                                            | 6       |
| `Winograd/Base/gfx1201/fp16/shadersBinReloc.hpp`                                            | 6       |
| `Winograd/Base/gfx1201/fp32/shadersBinReloc.hpp`                                            | 6       |
| `Winograd/Fury/gfx1100/fp16/shadersBinReloc.hpp`                                            | 4       |
| `Winograd/Fury/gfx1201/fp16/shadersBinReloc.hpp`                                            | 3       |
| `Winograd/Rage/gfx1201/fp16/shadersBinReloc.hpp`                                            | 2       |
| **Total**                                                                                   | **33**  |

Each output entry differs from its source only in:
1. the `StaticShaderType<SIZE>` template argument (linked size, generally
   smaller than the relocatable);
2. the symbol name (suffixed with `_NonReloc`);
3. the `.m_binary` byte block;
4. `.m_isRelocatable = true` becomes `.m_isRelocatable = false`.

Surrounding text (copyright header, namespace, includes, per-kernel
`// ...` comments, trailing field initializers) is preserved verbatim.

## Toolchain prerequisites

* **Python 3.10+** (uses dataclass kw-only attributes and PEP 604 unions).
* **`ld.lld`** with the AMDGPU backend. ROCm-for-Windows ships one at
  `C:\opt\rocm\bin\ld.lld.exe`; on Linux any ROCm install or a vanilla
  LLVM build configured with `-DLLVM_TARGETS_TO_BUILD="AMDGPU;X86"` works.
* **`llvm-readelf` or `llvm-readobj`** (for the post-link header
  assertions). ROCm-for-Windows only ships `llvm-readobj`; the link
  scripts auto-fall back to it and pass `--elf-output-style=GNU` to
  produce the GNU-style header dump they parse.

If any tool is missing, the link scripts exit with an actionable error
that names what they tried to find.

## How to run it

End-to-end, from the repository root:

```powershell
# Windows
powershell -NoProfile -ExecutionPolicy Bypass -File tools\winograd_pal\run.ps1
```

```bash
# Linux / WSL / Git Bash
tools/winograd_pal/run.sh
```

Both wipe `<repo>/build/winograd_pal/`, then run extract -> link -> reemit
in order. Re-running is idempotent.

The orchestrator forwards optional flags:

| Flag (`run.ps1` / `run.sh`)   | Purpose                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `-Lld` / `--lld`              | Explicit path to `ld.lld`                                 |
| `-Readelf` / `--readelf`      | Explicit path to `llvm-readelf` or `llvm-readobj`         |
| `-OnlyKernel` / `--only`      | Smoke-test gate: link a single kernel, skip re-emission   |
| `-BuildRoot` / `--build-root` | Override the `build/winograd_pal` output directory         |

## Smoke-test gate (run this first)

Before producing all 33 headers, the plan calls for linking exactly one
binary and verifying the linker output by hand. The orchestrator's
`-OnlyKernel` flag skips re-emission so it's safe to run repeatedly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\winograd_pal\run.ps1 `
    -OnlyKernel ConvWinogradElf_Gfx12_F3x2_Fp16Dot2Stride2Dec
```

Then inspect the linked output (the link scripts already assert most of
the below, but a manual look is cheap):

```powershell
$elf = "build\winograd_pal\Base\gfx1201\fp16\ConvWinogradElf_Gfx12_F3x2_Fp16Dot2Stride2Dec.linked.elf"
& "C:\opt\rocm\bin\llvm-readobj.exe" --elf-output-style=GNU -h $elf
& "C:\opt\rocm\bin\llvm-readobj.exe" --elf-output-style=GNU --notes $elf | Select-String "amdpal"
```

You should see:

* `Type: DYN (Shared object file)`
* `OS/ABI: AMDGPU - PAL`
* `Machine: EM_AMDGPU`
* `Flags: 0x48` (or whatever mach the source `.o` carried)
* the `amdpal.pipelines:` and `amdpal.version:` keys in the `.note`
  metadata dump

The hardware-side smoke test (side-loading the linked ELF through the
convmxn_base sample so `pipelineCrossCompileHip` can chew on it) is
out of scope for this tool but is the next thing to do once the file
inspection above looks right.

## How the pipeline fits together

```
shadersBinReloc.hpp (x7)
        |
        v
extract.py
   - regex-parses each `.m_binary = { 0x.., ... }` block
   - decodes -> raw bytes
   - patches `.symtab` `sh_info` (see "ELF defects in the source binaries")
   - writes <NAME>.o + manifest.json per directory under build/winograd_pal/
        |
        v
link.ps1 / link.sh
   - resolves ld.lld + llvm-readelf|llvm-readobj from PATH (or explicit args)
   - per kernel:
       ld.lld --shared -m elf64_amdgpu --no-undefined -Bsymbolic <NAME>.o -o <NAME>.linked.elf
   - patches OS/ABI byte of the linked ELF back to AMDGPU_PAL (0x41)
   - asserts: ET_DYN, OS/ABI=AMDGPU_PAL, EM_AMDGPU,
              e_flags mach byte unchanged from source
        |
        v
reemit.py
   - re-parses the source .hpp, identifies each kernel's text span
   - swaps in: new size, _NonReloc symbol suffix, new bytes,
               `.m_isRelocatable = false`
   - writes shadersBinNonReloc.hpp next to each shadersBinReloc.hpp
```

## ELF defects in the source binaries (and why we patch them)

The Winograd PAL pipeline ELFs ship with `sh_info = 0` on their `.symtab`
section header. Per the ELF spec, `sh_info` for a `SYMTAB` section must be
the index of the first non-`STB_LOCAL` symbol. Both `STN_UNDEF` (entry 0)
and `_amdgpu_cs_main` (entry 1) are `STB_LOCAL`, so the correct value is
`2`. LLVM's `ld.lld` rejects the malformed input with
`invalid sh_info in symbol table`, so `extract.py` fixes the field
in-place when it writes the `.o` files. The patch is recorded as
`sh_info_patched: true` in the per-kernel `manifest.json` entry. No
symbols, instructions, or section payloads are touched.

## Why these particular `ld.lld` flags

| Flag                | Why                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--shared`          | Emit `ET_DYN`, the whole point of the tool.                                                                                    |
| `-m elf64_amdgpu`   | Force the AMDGPU emulation; without it LLD picks the host emulation.                                                          |
| `--no-undefined`    | The PAL pipelines are self-contained, so any unresolved reference is a defect.                                                 |
| `-Bsymbolic`        | Bind references to local definitions, mirroring `comgr`'s behavior on the HSA path.                                            |
| (no `--gc-sections`)| `_amdgpu_cs_main` is `STB_LOCAL`, so GC would treat its `.text` section as unreachable and emit a 2.5 KB stub with no kernel. |
| (no `--pie`)        | ROCm's `ld.lld` rejects `--pie` together with `--shared`.                                                                       |

## Why we patch OSABI back to PAL after linking

`ld.lld --shared` always writes `OS/ABI = AMDGPU_HSA (0x40)` in
`e_ident[7]` regardless of the input OSABI. The embedded
`amdpal.pipelines` metadata in `.note` is preserved verbatim, but
downstream consumers may inspect the byte to dispatch HSA vs PAL
runtime paths. `link.ps1` / `link.sh` rewrite that single byte back
to `0x41 (AMDGPU_PAL)` after each successful link. No other byte
changes.

## Known caveats

* **`ld.lld` and PAL pipelines.** LLD's `--shared` flow is geared toward
  LLPC's pipeline assembly, not toward hand-assembled PAL relocatables.
  The output passes every static check we can apply (`ET_DYN`, OS/ABI,
  preserved metadata, preserved `.text`), but the only definitive test
  is to dispatch the linked binary through the actual D3D12 path. That
  hardware-side validation is intentionally out of scope for this tool;
  see the parent plan for follow-up work.
* **Multi-arch directories.** The `gfx1100` directory contains binaries
  for multiple Navi3x mach codes (`0x41 = gfx1100`, `0x47 = gfx1102`).
  The link scripts assert that the linked output's mach byte equals the
  source `.o`'s mach byte (rather than hardcoding allowed sets) so this
  works automatically.
* **Re-emission preserves only what it can locate.** The reemit step
  uses brace-balanced parsing, not a full C++ frontend. If a future
  edit to `shadersBinReloc.hpp` adds nested braces inside `.m_binary` (or
  reorders the standard initializer fields), the parser will throw
  with a kernel name and offset rather than silently produce a wrong
  output.
* **Encoder round-trip self-test.** Before touching any output
  `shadersBinNonReloc.hpp`, `reemit.py` decodes the first kernel of
  each source, re-encodes it through `_encode_bytes_block`, decodes
  again, and asserts equality. This catches encoder regressions before
  they ship.

## Out of scope (intentionally)

These belong to follow-up plans:

* Wiring the new `_NonReloc` symbols into `selectRelocatableShader()` /
  `getShadersBlob()` for Base/Fury/Rage so the convmxn flow uses them
  in place of the comgr `LINK_RELOCATABLE_TO_EXECUTABLE` path.
* CMake changes to compile the new headers in.
* Reverting the temporary `sourceArchForTarget()` hack in the Winograd
  shader utilities (becomes moot once we no longer go through
  `getNonRelocatable()`).
* End-to-end re-runs of the unit-test and convmxn-sample suites.
