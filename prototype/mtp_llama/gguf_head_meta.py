#!/usr/bin/env python3
"""Read GGUF header metadata (key-value section only) for the MTP fixture used by the
llama.cpp A/B harness. CPU-only; does not touch the GPU or read tensor data."""
import struct
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "tmp/mtp_llama/Qwen3.5-0.8B.mtp.bf16.gguf"

# GGUF v3 type ids: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 str 9 arr 10 u64 11 i64 12 f64
FM = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
      5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def rstr(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        cnt = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            return [rstr(f) for _ in range(cnt)]
        fmt, nb = FM[et]
        return [struct.unpack(fmt, f.read(nb))[0] for _ in range(cnt)]
    fmt, nb = FM[t]
    return struct.unpack(fmt, f.read(nb))[0]


def main():
    with open(PATH, "rb") as f:
        assert f.read(4) == b"GGUF", "not a gguf"
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        meta = {}
        for _ in range(n_kv):
            k = rstr(f)
            t = struct.unpack("<I", f.read(4))[0]
            meta[k] = rval(f, t)
    print(f"version={ver} n_tensors={n_tensors} n_kv={n_kv}")
    for k, v in sorted(meta.items()):
        sv = str(v)
        if len(sv) > 120:
            sv = sv[:117] + "..."
        print(f"{k} = {sv}")


if __name__ == "__main__":
    main()
