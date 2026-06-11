#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Drive ld.lld to link every Winograd PAL relocatable extracted by
# extract.py into a shared (ET_DYN) AMDPAL ELF. Mirror of link.ps1; same
# logic, different shell.
#
# Usage:
#   tools/winograd_pal/link.sh [--build-root DIR] [--lld PATH]
#                              [--readelf PATH] [--only KERNEL]
#
# A non-zero exit indicates either a missing tool, a failed link, or a
# linked ELF that did not pass the post-link sanity checks.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

build_root="${repo_root}/build/winograd_pal"
lld=""
readelf=""
only_kernel=""

usage() {
    sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-root) build_root="$2"; shift 2 ;;
        --lld)        lld="$2";        shift 2 ;;
        --readelf)    readelf="$2";    shift 2 ;;
        --only)       only_kernel="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

resolve_tool() {
    local override="$1"
    local friendly="$2"
    shift 2
    if [[ -n "${override}" ]]; then
        if [[ ! -x "${override}" ]]; then
            echo "ERROR: ${friendly} not executable at: ${override}" >&2
            exit 1
        fi
        printf '%s\n' "${override}"
        return
    fi
    for cand in "$@"; do
        if command -v "${cand}" >/dev/null 2>&1; then
            command -v "${cand}"
            return
        fi
    done
    cat >&2 <<EOF
ERROR: ${friendly} not found on PATH (looked for: $*).
Install ROCm or build LLVM with LLVM_TARGETS_TO_BUILD=AMDGPU, then either
prepend its bin/ to PATH or pass an explicit path.
EOF
    exit 1
}

read_elf_header() {
    # Both `llvm-readelf` and `llvm-readobj` accept `--elf-output-style=GNU`
    # and produce the GNU-style ELF header dump used by assert_linked_elf.
    "$1" --elf-output-style=GNU -h "$2"
}

# ELFOSABI_AMDGPU_PAL = 0x41 at e_ident[7]. Restore it after ld.lld --shared
# (which always emits 0x40 / HSA), since the .note metadata still describes
# a PAL pipeline.
set_pal_osabi() {
    local elf="$1"
    local kernel="$2"
    if [[ ! -f "${elf}" ]]; then
        echo "ERROR: ${kernel}: linked file missing: ${elf}" >&2
        exit 1
    fi
    local magic
    magic="$(head -c 4 "${elf}" | xxd -p)"
    if [[ "${magic}" != "7f454c46" ]]; then
        echo "ERROR: ${kernel}: linked file is not an ELF" >&2
        exit 1
    fi
    printf '\x41' | dd of="${elf}" bs=1 seek=7 count=1 conv=notrunc status=none
}

amdgpu_mach_byte() {
    # ELF64 e_flags lives at offset 0x30; AMDGPU mach code is the low byte.
    local elf="$1"
    if [[ ! -f "${elf}" ]]; then
        echo "ERROR: cannot read mach byte from missing file: ${elf}" >&2
        exit 1
    fi
    dd if="${elf}" bs=1 skip=48 count=1 status=none 2>/dev/null | xxd -p
}

assert_linked_elf() {
    local readelf_path="$1"
    local elf="$2"
    local source_obj="$3"
    local kernel="$4"

    local out
    if ! out="$(read_elf_header "${readelf_path}" "${elf}" 2>&1)"; then
        echo "ERROR: readelf -h failed for ${kernel}:" >&2
        echo "${out}" >&2
        exit 1
    fi

    if ! grep -qE 'Type:[[:space:]]+DYN' <<<"${out}"; then
        echo "ERROR: ${kernel}: expected Type: DYN (Shared object)" >&2
        echo "${out}" >&2
        exit 1
    fi
    if ! grep -qE 'OS/ABI:[[:space:]]+AMDGPU[[:space:]]*-[[:space:]]*PAL' <<<"${out}"; then
        echo "ERROR: ${kernel}: expected OS/ABI: AMDGPU - PAL" >&2
        echo "${out}" >&2
        exit 1
    fi
    if ! grep -q 'EM_AMDGPU' <<<"${out}"; then
        echo "ERROR: ${kernel}: expected Machine: EM_AMDGPU" >&2
        echo "${out}" >&2
        exit 1
    fi

    local flags_hex
    flags_hex="$(grep -oE 'Flags:[[:space:]]+0x[0-9a-fA-F]+' <<<"${out}" \
                 | sed -E 's/.*0x//' | tr 'A-F' 'a-f' | head -n1)"
    if [[ -z "${flags_hex}" ]]; then
        echo "ERROR: ${kernel}: could not parse Flags from readelf output" >&2
        exit 1
    fi
    local linked_mach source_mach
    linked_mach="$(printf '%02x' "$((16#${flags_hex} & 0xFF))")"
    source_mach="$(amdgpu_mach_byte "${source_obj}")"

    if [[ "${linked_mach}" != "${source_mach}" ]]; then
        echo "ERROR: ${kernel}: e_flags mach changed during link (source=0x${source_mach}, linked=0x${linked_mach})" >&2
        exit 1
    fi
}

link_manifest() {
    local lld_path="$1"
    local readelf_path="$2"
    local manifest="$3"
    local only="$4"

    local dir
    dir="$(dirname "${manifest}")"

    # Pull family/arch/precision and per-kernel records via python (avoids
    # forcing a jq dependency on developer machines).
    local kernels
    kernels="$(python - <<EOF
import json, sys
m = json.load(open(r"${manifest}"))
print(m["family"])
print(m["arch"])
print(m["precision"])
for k in m["kernels"]:
    print(f"{k['name']}\t{k['dst_o']}")
EOF
)"
    local family arch precision
    family="$(sed -n '1p' <<<"${kernels}")"
    arch="$(sed -n '2p' <<<"${kernels}")"
    precision="$(sed -n '3p' <<<"${kernels}")"

    local count=0
    while IFS=$'\t' read -r name rel_o; do
        [[ -z "${name}" ]] && continue
        if [[ -n "${only}" && "${name}" != "${only}" ]]; then continue; fi

        local obj_basename out log obj
        obj_basename="$(basename "${rel_o}")"
        obj="${dir}/${obj_basename}"
        out="${obj%.o}.linked.elf"
        log="${obj%.o}.link.log"
        # arch is read out of the manifest a few lines above; keep used for
        # output formatting only.
        : "${arch}"

        if [[ ! -f "${obj}" ]]; then
            echo "ERROR: missing extracted object: ${obj}" >&2
            exit 1
        fi

        echo "  linking ${family}/${arch}/${precision}/${name}"
        # Notes on the flag set:
        #   --shared          emit ET_DYN (the whole point of this tool).
        #   --no-undefined    treat unresolved references as a hard error.
        #   -Bsymbolic        bind references to local definitions.
        # `--gc-sections` is intentionally omitted: the kernel entry point
        # `_amdgpu_cs_main` is a LOCAL symbol, which GC would treat as
        # unreachable and drop along with its .text section.
        # `--pie` is intentionally omitted: ROCm's ld.lld treats it as
        # incompatible with `--shared`.
        if ! "${lld_path}" --shared -m elf64_amdgpu --no-undefined \
                            -Bsymbolic \
                            "${obj}" -o "${out}" >"${log}" 2>&1; then
            echo "ERROR: ld.lld failed for ${name}; see ${log}" >&2
            exit 1
        fi

        # Restore OSABI=AMDGPU_PAL on the linked output (LLD always writes
        # 0x40/HSA for --shared, but the embedded amdpal.pipelines metadata
        # in .note still identifies this as a PAL pipeline).
        set_pal_osabi "${out}" "${name}"

        assert_linked_elf "${readelf_path}" "${out}" "${obj}" "${name}"
        count=$((count + 1))
    done < <(tail -n +4 <<<"${kernels}")
    echo "${count}"
}

lld_path="$(resolve_tool "${lld}" "ld.lld" ld.lld)"
readelf_path="$(resolve_tool "${readelf}" "llvm-readelf/llvm-readobj" llvm-readelf llvm-readobj)"

echo "ld.lld:       ${lld_path}"
echo "llvm-readelf: ${readelf_path}"
echo "build root:   ${build_root}"

if [[ ! -d "${build_root}" ]]; then
    echo "ERROR: build root does not exist: ${build_root}. Run extract.py first." >&2
    exit 1
fi

mapfile -t manifests < <(find "${build_root}" -name manifest.json -type f | sort)
if [[ ${#manifests[@]} -eq 0 ]]; then
    echo "ERROR: no manifest.json found under ${build_root}. Run extract.py first." >&2
    exit 1
fi

total=0
for m in "${manifests[@]}"; do
    n="$(link_manifest "${lld_path}" "${readelf_path}" "${m}" "${only_kernel}")"
    total=$((total + n))
done

echo
if [[ -n "${only_kernel}" ]]; then
    echo "Linked ${total} kernel(s) matching name '${only_kernel}'."
else
    echo "Linked ${total} kernel(s) successfully."
fi
