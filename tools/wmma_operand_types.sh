#!/usr/bin/env bash
# Which WMMA operand types does this toolchain actually emit matrix instructions for on gfx1151?
#
# Why this exists: `devices/strix-halo-compute.md` and `devices/strix-halo-official-specs.md` recorded only
# `v_wmma_f32_16x16x16_f16`, so every plan for a matrix-path attention assumed the bf16 operands would have to
# be CONVERTED to fp16 first - a numerics change under the standing 16-bit authority. Whether that premise is
# even true is a compile question with a 30-second answer, and nobody had taken it.
#
# Usage: tools/wmma_operand_types.sh          # both variants
#          VARIANT=0 tools/wmma_operand_types.sh   # bf16 only
# Nothing here touches the device: the question is what the compiler emits for `--offload-arch=gfx1151`, so this
# runs in any window and needs no lock. A variant that "compiles" but lowers to something other than a matrix
# instruction is reported as such, because accepting the builtin and then emitting a scalar loop would look
# identical at compile time and is exactly the trap this probe is built to avoid.
set -u
cd "$(dirname "$0")/.."
SRC="$PWD/tools/wmma_operand_types.cpp"
OUT="$PWD/tmp/wmma_operand_types"
mkdir -p "$OUT"
cd "$OUT"
# `--save-temps` writes its intermediates into the CURRENT directory, so the compile runs here and the listing
# has this exact name; a tool that greps for a file it did not produce must say so rather than report a miss as
# an absence of the instruction.
LISTING="$OUT/$(basename "$SRC" .cpp)-hip-amdgcn-amd-amdhsa-gfx1151.s"

variants=(0 1)
[ -n "${VARIANT:-}" ] && variants=("$VARIANT")
rc=0
for v in "${variants[@]}"; do
    case "$v" in
        0) name="bf16" ;;
        1) name="f16"  ;;
        *) echo "VARIANT=$v unknown (0=bf16, 1=f16)"; exit 2 ;;
    esac
    printf 'gfx1151 %s wmma: ' "$name"
    # Delete the previous variant's listing first. `--save-temps` writes one fixed filename for both variants,
    # so without this the second variant greps the FIRST variant's assembly and both rows come out identical -
    # which is how the first version of this tool reported bf16 as `v_wmma_f32_16x16x16_f16`.
    rm -f "$LISTING"
    if ! hipcc -O2 -DVARIANT="$v" --offload-arch=gfx1151 --save-temps -c "$SRC" -o "$OUT/v$v.o" 2>"$OUT/v$v.err"; then
        echo "REFUSED at compile: $(grep -m1 'error:' "$OUT/v$v.err" | cut -c1-120)"
        rc=1
        continue
    fi
    if [ ! -s "$LISTING" ]; then
        echo "no listing was produced ($LISTING missing) - the check cannot be made"
        rc=1
        continue
    fi
    ins=$(grep -ohE 'v_wmma[a-z0-9_]*|v_mfma[a-z0-9_]*' "$LISTING" | sort -u | tr '\n' ' ')
    if [ -n "$ins" ]; then
        echo "PRESENT -> $ins"
    else
        echo "no matrix instruction in the listing (the builtin was accepted and lowered to something else)"
        rc=1
    fi
done
exit $rc
