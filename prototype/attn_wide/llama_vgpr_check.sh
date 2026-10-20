#!/usr/bin/env bash
# Re-measure llama.cpp's RDNA3 flash-attention register reservation and spill with the CORRECTED instrument.
#
#   bash prototype/attn_wide/llama_vgpr_check.sh [outdir]        # one clang++ compile, CPU only, ~1-2 min
#   WANT=ILi128ELi128ELi16ELi4ELb0ELb0ELb0 bash ...              # any other instantiation of the same template
#
# WHY THIS EXISTS. A12's spill budget — "≤ 1024 B/thread", and with it the whole "256 VGPR plus a relaxed spill
# policy is a funded shape, not a gamble" premise — is anchored on llama's kernel: `A2-census.md`'s row
# `flash_attn_ext_f16<256,256,16,4,false,false,false>` reads "256 VGPR, 900 B/thr". That census predates the
# instrument correction in `prototype/attn_percost/A11-attribution.md` §2 and AGENTS.md's note-block row, which
# together say a `.vgpr_count` read out of LLVM's note block can report another kernel's number, because note
# entries are not in definition order. So the anchor number had to be re-taken from the code-object header:
# `.amdhsa_next_free_vgpr` is what the assembler emits and the KFD reserves, and
# `.amdhsa_private_segment_fixed_size` is what the driver hands out per thread as scratch.
#
# WHAT IS READ, and what is deliberately not:
#   .amdhsa_next_free_vgpr               the register reservation              → authoritative
#   .amdhsa_private_segment_fixed_size   scratch (spill) bytes per thread      → authoritative
#   .amdhsa_group_segment_fixed_size     STATIC LDS only. llama stages with `extern __shared__`, so this column
#                                        reads 0 for it and the real figure is derived from the source and
#                                        recorded in `A12-WIDE-STEP12.md`. A 0 here is "dynamic", not "none".
#   scratch_{load,store} sites           a behavioural witness that the spill is real: one site is one per thread,
#                                        which is the unit the budget is written in.
# The note block's `.vgpr_spill_count` is NOT printed: pairing that field with a kernel name is the exact mistake
# AGENTS.md records, and the fields above need no help.
#
# FIDELITY. The compile reuses llama's OWN flags, read verbatim out of the CMake-generated flags file for the
# object that contains this instantiation, and compiles llama's own instance translation unit rather than a
# hand-written look-alike. The instance file is FOUND by searching the tree for the mangled name, because the
# filename-to-arguments mapping is not guessable: `<256,256,16,4,false,false,false>` lives in
# `fattn-mma-f16-instance-ncols1_16-ncols2_4.cu`, which reads like sixteen columns.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:-$repo/tmp/attn_wide_llama}
want=${WANT:-ILi256ELi256ELi16ELi4ELb0ELb0ELb0}     # mangled template arguments; A2-census's row is the default
llvmbin=${LLVM_BIN:-/opt/rocm/lib/llvm/bin}
mkdir -p "$out"; cd "$out"

inst=$repo/engines/llama.cpp/ggml/src/ggml-cuda/template-instances
flags=$repo/engines/llama.cpp/build-hip/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/flags.make
[ -d "$inst" ] || { echo "ABORT: $inst is not in the tree (llama source missing)"; exit 1; }
[ -f "$flags" ] || { echo "ABORT: $flags is missing, so llama's own build flags cannot be read"; exit 1; }

# Which shipped object already carries the instantiation, found by symbol search: the filename-to-arguments
# mapping is not guessable (`<256,256,16,4,false,false,false>` is `ncols1_16-ncols2_4`), and the CMake layout maps
# an object back onto its translation unit one suffix at a time. Both halves are asserted, so a missing build tree
# or a missing source says which one is missing instead of compiling the wrong file.
obj=""
for f in $(find "$repo/engines/llama.cpp/build-hip" -name 'fattn-mma-f16-instance-*.o' 2>/dev/null | sort); do
  # The mangled name is `..._18flash_attn_ext_f16ILi256E...`: the template argument list is glued straight onto the
  # name, so the substring to match is the mangled-argument string alone, with no angle brackets in it. And NO
  # `llvm-nm -g`: llama declares these kernels `static` — the `L` in `_ZL18flash_attn_ext_f16` — so a
  # globals-only dump cannot see the one kernel this script exists to measure, and would report "no object defines
  # it" about an object that does.
  if "$llvmbin/llvm-nm" --defined-only "$f" 2>/dev/null | grep -q "$want"; then obj=$f; break; fi
done
[ -n "$obj" ] || { echo "ABORT: no object under engines/llama.cpp/build-hip defines flash_attn_ext_f16<$want>"; exit 1; }
src=${obj%.o}
src=$repo/engines/llama.cpp/ggml/src/ggml-cuda/template-instances/$(basename "$src")
[ -f "$src" ] || { echo "ABORT: object $obj maps to $src, which does not exist"; exit 1; }

echo "instantiation              : flash_attn_ext_f16<$want>"
echo "shipped object holding it  : $obj"
echo "instance TU compiled       : $src"

# Word-splitting on purpose: flags.make holds each variable as one flat space-separated line and no flag in this
# build contains a space. Re-quoting them would be the thing that breaks.
# shellcheck disable=SC2086
/opt/rocm/lib/llvm/bin/clang++ -x hip \
    $(sed -n 's/^HIP_FLAGS = //p' "$flags") \
    $(sed -n 's/^HIP_DEFINES = //p' "$flags") \
    $(sed -n 's/^HIP_INCLUDES = //p' "$flags") \
    --save-temps -c "$src" -o "$out/inst.o" > "$out/compile.log" 2>&1 \
    || { echo "ABORT: compile failed, tail of $out/compile.log:"; tail -5 "$out/compile.log"; exit 1; }
listing=$(ls "$out"/*amdgcn-amd-amdhsa-gfx1151.s 2>/dev/null | head -1)
[ -n "$listing" ] || { echo "ABORT: --save-temps produced no GCN listing"; exit 1; }
echo "listing read               : $listing"
echo

# ONE pass, keyed by the mangled name, which is identical in the function-body label and in the `.amdhsa_kernel`
# block — so a field can only ever be reported against the kernel it belongs to. This is the property the
# note-block parser lacked.
awk -v want="$want" '
  /^[^.[:space:]#;].*:/ {                                 # function body label at column 0, and NOT a `.LBB`/
      f = $0; sub(/:.*/, "", f)                          # `.Lfunc_end` label — matching those too is how the
      if (index(f, want)) { cur = f; sites[f] += 0; if (!(f in seen)) { seen[f] = ++n; order[n] = f } }
      else cur = ""
      next
  }
  /^[[:space:]]*\.amdhsa_kernel[[:space:]]/ {
      k = $2
      if (index(k, want)) { blk = k; if (!(k in seen)) { seen[k] = ++n; order[n] = k }
                            vgpr[k] = "?"; priv[k] = "?"; grp[k] = "?"; sites[k] += 0 }
      else blk = ""
      next
  }
  blk != "" && /\.amdhsa_next_free_vgpr/                 { vgpr[blk] = $2 }
  blk != "" && /\.amdhsa_private_segment_fixed_size/     { priv[blk] = $2 }
  blk != "" && /\.amdhsa_group_segment_fixed_size/       { grp[blk]  = $2 }
  cur != "" && /scratch_(load|store)/                   { sites[cur]++ }
  END {
      printf "%-52s %6s %7s %10s %7s %14s\n", "kernel", "vGPU", "alloc8", "scratch B", "LDS B", "scratch sites"
      for (i = 1; i <= n; i++) {
          k = order[i]
          a = (vgpr[k] == "?" ? "?" : int((vgpr[k] + 7) / 8) * 8)
          printf "%-52s %6s %7s %10s %7s %14d\n", k, vgpr[k], a, priv[k], grp[k], sites[k]
      }
      if (n == 0) print "NO MATCH — the listing does not contain the requested instantiation"
  }
' "$listing"

echo
echo "Against A2-census: 256 VGPR and 900 B/thread. vGPU is the reservation the KFD makes at the per-thread ceiling"
echo "of 256; alloc8 is it rounded at the driver's granularity of 8; scratch B is the per-thread private segment."
echo "LDS B is 0 for llama because its stage is extern __shared__ (dynamic), not because there is none."
