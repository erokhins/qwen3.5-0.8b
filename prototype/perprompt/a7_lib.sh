# a7_lib.sh — the parsers `a7_host_split.sh` reads run logs with, split out so they can be
# self-tested against fixture lines without touching the device (`a7_parse_selftest.sh`).
#
# The rule these implement: a phase that produced no number is a FAILURE, never a zero. Every
# reader returns a bare number or calls `die`, which is the trap `AGENTS.md` records twice
# (`gdn_collapse/run.py`'s zero-match regex printing PASS; `gate.py`'s 1023-missing-rows run
# reporting a numerics verdict). A `kv` that silently returned "" would turn a missing
# `WAVEPASS` line into a "0 ms warm wave" and a 147 ms gap out of nothing.
#
# Sourced, not executed: `die` must exit the CALLER, so the caller defines `die` first.

# kv FILE KEY LABEL — value of `KEY=` on the last line that carries it.
kv() {
    local f=$1 key=$2 label=$3 v
    v=$(grep -oE "${key}=[0-9]+(\.[0-9]+)?" "$f" | tail -1 | cut -d= -f2)
    [ -n "$v" ] || die "field '$label' (key ${key}) parsed EMPTY from $f — line missing, not zero"
    printf '%s' "$v"
}

# wave_wall FILE GRAN LABEL — the `wall_ms` of one `WAVEPASS gran=<GRAN>` line, which is what
# §49 subtracts. `wave_pass` prints exactly one per granularity (`engine/decode_run.cpp:1652`).
wave_wall() {
    local f=$1 gran=$2 label=$3 v
    v=$(grep "^WAVEPASS gran=${gran} " "$f" | head -1 | sed -E 's/.*wall_ms=([0-9.]+).*/\1/')
    [ -n "$v" ] || die "no 'WAVEPASS gran=${gran} wall_ms=' line in $f ($label): the warm re-pass did not run, so there is nothing to subtract"
    printf '%s' "$v"
}

# load_ms FILE — prose, not key=value: `loaded 335 tensors, 1.5457 GB resident in 524.3 ms
# (335 allocations)` (`engine/decode_run.cpp:1804`). This is the wall the headline excludes.
load_ms() {
    local f=$1 v
    v=$(sed -n 's/^loaded .*resident in \([0-9.]*\) ms.*/\1/p' "$f" | tail -1)
    [ -n "$v" ] || die "no 'loaded … resident in X ms' line in $f — the load wall is unattributed"
    printf '%s' "$v"
}

# med — median of the numbers on stdin. A blank line is NOT a rep: non-numeric input is
# dropped and an empty list is an error, because a median over "nothing" printed as 0 is the
# same class of lie as a missing warm wall read as 0 ms.
med() {
    local v
    v=$(grep -E '^[+-]?[0-9]+(\.[0-9]+)?$')
    [ -n "$v" ] || { echo "med: EMPTY rep list" >&2; return 1; }
    printf '%s\n' "$v" | sort -g | awk '{a[NR]=$1} END {
        print (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2 }'
}

# diff_ms A B — A minus B with a fixed 3 decimals, so a median of gaps stays parseable.
diff_ms() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", a - b}'; }
