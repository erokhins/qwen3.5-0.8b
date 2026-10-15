#!/usr/bin/env bash
# device_facts.sh — every static fact about this box that devices/*.md cites, in one diffable block.
#
# Why this exists: the repo's rooflines rest on device facts (40 CUs, 2 MiB L2, 237 GB/s, the ~34 GiB
# pool, the sclk window), and three of them turned out to disagree with each other or with the vendor's
# page. A future session should be able to diff "the box today" against "the box the doc was written
# on" without retyping a dozen commands and without needing a network:
#
#     diff <(git show HEAD:devices/device-facts.txt) <(tools/device_facts.sh --pairs)
#
# STATIC facts only — no allocation, no kernel launch, no timing — so it is safe to run while a
# subagent holds the GPU. Bandwidth, compute and occupancy are separate device measurements that need
# a clean window: tools/mem_bw, tools/mixed_bw, tools/compute_bw, tools/prefill_occupancy.py,
# tools/prefill_kernel_trace.py.
#
# LABEL TRAP, learned the hard way: `rocminfo` prints one block per agent and EVERY block has a
# "Compute Unit:" line. The CPU agent's says 32 (that is 32 SMT threads), the 8060S agent's says 40,
# the NPU's says 0. Flattening all three under one prefix makes the CPU's 32 look like a GPU CU count
# that contradicts AMD's published "Graphics Core Count 40". Every row below is prefixed with the
# agent it came from; the number to compare with the vendor page is `gfx1151.ComputeUnit=40`.
set -uo pipefail

PAIRS=0
[ "${1:-}" = "--pairs" ] && PAIRS=1
kv() { if [ "$PAIRS" = 1 ]; then printf '%s=%s\n' "$1" "$2"; else printf '%-30s %s\n' "$1" "$2"; fi; }
sec() { [ "$PAIRS" = 1 ] || printf '\n## %s\n' "$1"; }

sec "identity"
for f in sys_vendor product_name product_family product_sku board_name board_vendor \
         bios_version bios_date; do
  kv "dmi.$f" "$(cat "/sys/class/dmi/id/$f" 2>/dev/null || echo -)"
done

sec "cpu (lscpu)"
lscpu 2>/dev/null | while IFS= read -r line; do
  case "$line" in
    "Model name"*|"CPU(s):"*|"Thread(s) per core:"*|"Core(s) per socket:"*|"Socket(s):"*|\
    "CPU max MHz"*|"CPU min MHz"*|"L1d cache"*|"L1i cache"*|"L2 cache"*|"L3 cache"*|"NUMA node(s):"*)
      kv "lscpu.$(echo "${line%%:*}" | tr -d ' ()')" "$(echo "${line#*:}" | xargs)" ;;
  esac
done

sec "memory as the kernel sees it"
kv MemTotal_kB "$(awk '/MemTotal/{print $2}' /proc/meminfo)"
kv MemAvailable_kB "$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
kv kernel "$(uname -r)"
kv cmdline_amdgpu "$(tr ' ' '\n' < /proc/cmdline | grep '^amdgpu' | tr '\n' ' ')"
kv gtt_MiB "$(tr ' ' '\n' < /proc/cmdline | sed -n 's/^amdgpu\.gttsize=//p')"

sec "agents as rocminfo sees them (cpu_agent | gfx1151 | npu)"
rocminfo 2>/dev/null | awk '
  /^[0-9]+\]|^\[[0-9]+\]/ { }
  /Marketing Name:/ {
    n = $0; sub(/.*Marketing Name: */, "", n); sub(/ *$/, "", n)
    if      (n ~ /8060S Graphics/) a = "gfx1151"
    else if (n ~ /npu|NPU/)        a = "npu"
    else if (n ~ /RYZEN|Ryzen/)    a = "cpu_agent"
    else                           a = "other"
    print a "|MarketingName|" n; next
  }
  a != "" {
    k = $0; sub(/:.*/, "", k); gsub(/[ ()\/-]/, "", k)
    if (k ~ /^(ComputeUnit|SIMDsperCU|WavefrontSize|MaxWavesPerCU|MaxWorkitemPerCU|MaxClockFreqMHz|L1|L2)$/) {
      v = $0; sub(/.*: */, "", v); gsub(/ *$/, "", v); print a "|" k "|" v
    }
  }' | while IFS='|' read -r a k v; do kv "$a.$k" "$v"; done

sec "gpu clocks and memory as seen now (rocm-smi)"
rocm-smi --showclocks --showmeminfo vram 2>/dev/null | awk -F': ' '
  /^GPU\[/ {
    k = $2; v = $3
    for (i = 4; i <= NF; i++) v = v ": " $i
    gsub(/[ .()]/, ".", k); gsub(/^\.+|\.+$/, "", k)
    print k "|" v
  }' | while IFS='|' read -r k v; do kv "smi.$k" "$v"; done
kv sysfs.mem_info_vram_total "$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1)"
kv sysfs.mem_info_vram_used "$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)"

sec "software"
kv rocm_version "$(/opt/rocm/bin/hipcc --version 2>/dev/null | awk -F'version ' '/HIP version/{print $2; exit}')"
dpkg -l 2>/dev/null | awk '/^ii .*rocm-core/{print $2 "=" $3}'
kv llvm_target "$(/opt/rocm/bin/llvm-config --version 2>/dev/null)"
kv rocprofv3 "$(command -v rocprofv3 2>/dev/null || echo -)"
kv rocprofv3_version "$(rocprofv3 --version 2>/dev/null | head -1)"

if [ "$PAIRS" != 1 ]; then
  cat <<'EOT'

Static facts only. For the numbers that need a device and a clean window see:
  tools/mem_bw                  pure-read probe (the 237 GB/s denominator)
  tools/mixed_bw                mixed read/write and resident-set sweeps
  tools/compute_bw              matrix and FMA rate (the MFMA question in strix-halo-official-specs.md)
  tools/prefill_occupancy.py    busy% and sysfs sampling while a prefill runs
  tools/prefill_kernel_trace.py per-kernel attribution via rocprofv3 --kernel-trace
EOT
fi
