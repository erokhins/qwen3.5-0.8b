"""G3 census aid: the work table and the ceiling adjudication for the §38 GDN gap.

Everything here is arithmetic on quantities that are already measured and cited, in this repo, by
someone else — the point of putting it in a script is that the inputs are named once and the derived
figures cannot then drift from them. It measures nothing.

Inputs and where each comes from:
  * §38 traced times           progress/2026-10-21/performance-status-2026-10-21.md:1750  (gdn_recur: ours 272.07 ms,
                               llama 47.20 ms per pass of 4,096 lanes; 144 launches each)
  * our per-token bill         prototype/gdn_walk/evidence-tokenbill-2026-10-19.txt (token_bill.py:
                               1,327 slots/token-step/wavefront for rec_loop_kernel; BET.md §2)
  * llama's per-token bill     prototype/gdn_walk/g3_isa_bill.py over the code object carved out of
                               build-hip's gated_delta_net.cu.o: 213 instructions in the token loop
  * geometry                   prototype/gdn_walk/g3_trace_geom.py over §38's own traces
  * machine roofs              prototype/issue_rate_probe/results.md §1 (40 CU, 2.894 GHz, scalar
                               v_fma_f32 7.30 TFLOP/s) and devices/strix-halo-memory-bandwidth.md
                               §0b (918 GB/s at an 18 MiB working set = the GDN state itself),
                               §1 (237-240 GB/s DRAM plateau)

    python3 prototype/gdn_walk/g3_work_table.py
"""

D = 128                  # head_dim / state side (engine: D)
NH = 16                  # value heads (engine: NH)
LAYERS = 18              # GDN layers per pass: "18 x 4,096 launches became 18 x 8" decode_step.cpp:789
TOKENS = 4096            # lanes per pass
SPAN = 512               # token-steps per launch, both engines (ours --gdn-loop 512; llama -ub 512)
F4 = 4

STATE_EL = NH * D * D                     # 262,144
STATE_B = STATE_EL * F4                   # 1,048,576 B = 1 MiB

# --- geometry (g3_trace_geom.py over §38's CSVs) -----------------------------------------------
OURS = dict(threads=128, blocks=16, el_thread=128, slots_tok_wf=1327, ms=272.07)
LLAMA = dict(threads=128, blocks=128, el_thread=16, slots_tok_wf=213, ms=47.20)
for e in (OURS, LLAMA):
    e["wf"] = e["blocks"] * e["threads"] // 32
    e["threads_tot"] = e["blocks"] * e["threads"]
    assert e["wf"] * 32 * e["el_thread"] == STATE_EL, (e, "thread map must cover the state exactly")

# --- machine ceilings ---------------------------------------------------------------------------
SIMD = 80                # KFD simd_count (AGENTS.md: HIP's 20-CU view is halved)
GHZ = 2.894
ISSUE_ROOF = SIMD * GHZ * 1e9             # one wavefront slot per SIMD per cycle
SCALAR_FMA_TFLOPS = 7.30                  # measured sustained v_fma_f32
SCALAR_MAC_S = SCALAR_FMA_TFLOPS * 1e12 / 2
BW_CACHE_G = 918.0                          # GB/s at an 18 MiB working set (the GDN states)
BW_DRAM_G = 237.0                           # GB/s, cold DRAM plateau

TOK_LAYERS = LAYERS * TOKENS


def bill(e, state_global_passes, operand_bytes, redundant_operand_bytes=0, unique_bytes=0):
    wf = e["wf"]
    slots_tok = wf * e["slots_tok_wf"]
    slots_pass = slots_tok * TOK_LAYERS
    secs = e["ms"] / 1e3
    # arithmetic: 4 FMA-class ops per state element per token in BOTH spellings (see the doc)
    mac_tok = STATE_EL * 4
    # bytes demanded at the LSU: state passes + operands, per token
    bytes_tok = state_global_passes * STATE_B + operand_bytes + redundant_operand_bytes
    uniq_tok = unique_bytes
    return dict(
        wf=wf,
        slots_tok=slots_tok,
        Gslots=slots_pass / secs / 1e9,
        pct_issue=100 * (slots_pass / secs) / ISSUE_ROOF,
        TFLOPS=mac_tok * TOK_LAYERS / secs * 2 / 1e12,
        pct_fma=100 * (mac_tok * TOK_LAYERS / secs) / SCALAR_MAC_S,
        MB_tok=bytes_tok / 1e6,
        GBs=bytes_tok * TOK_LAYERS / secs / 1e9,
        MB_tok_unique=uniq_tok / 1e6,
        GBs_unique=uniq_tok * TOK_LAYERS / secs / 1e9,
    )


def main():
    print(f"state: {STATE_EL:,} elements = {STATE_B/2**20:.0f} MiB per layer; {LAYERS} GDN layers x "
          f"{TOKENS} tokens = {TOK_LAYERS:,} token-layers per pass")
    print(f"issue roof = {SIMD} SIMD x {GHZ} GHz = {ISSUE_ROOF/1e9:.1f} G wavefront-slots/s")
    print(f"scalar v_fma_f32 roof = {SCALAR_FMA_TFLOPS} TFLOP/s = {SCALAR_MAC_S/1e12:.2f} T MAC/s")
    print()
    # ours: 3 global state passes per token (loop A read, loop B read, loop B write); k,q,v,o,g,beta
    ours_oper = 3 * NH * D * F4 + NH * D * F4 + 2 * NH * F4          # 32,896 B
    # llama: the state is loaded once and stored once per SPAN, from registers. Its k/q rows are
    # re-read by every column group (4x inside a wavefront) and by every z-block (8x): that is L1
    # broadcast traffic, but it is real LSU demand and real instructions.
    ll_oper = 3 * NH * D * F4 + NH * D * F4 + 2 * NH * F4 + NH * D * F4   # k,q,v,o + g,beta
    ll_kq_redundant_per_tok = (4 * 8 - 1) * 2 * NH * D * F4          # beyond the unique row bytes
    ll_unique = 2 * NH * D * F4 + NH * D * F4 + NH * D * F4 + 2 * NH * F4  # k,q,v,o + g,beta
    b = {}
    # "unique" = distinct bytes touched per token, i.e. what cannot be absorbed by reuse inside the
    # token: ours reads the state twice and writes it once, so 2 MiB of distinct traffic per token
    # (the second read may hit L1); llama's state is touched twice per 512 tokens = 4 KiB/token.
    b["ours"] = bill(OURS, 3, ours_oper, 0, 2 * STATE_B + ours_oper)
    b["llama"] = bill(LLAMA, 2 / SPAN, ll_oper, ll_kq_redundant_per_tok,
                      (2 / SPAN) * STATE_B + ll_unique)
    for k, v in b.items():
        print(f"---- {k}")
        for kk, vv in v.items():
            print(f"   {kk:>10} = {vv:,.2f}")
    print()
    print("HEAD-TO-HEAD (per token per layer unless stated)")
    rows = [
        ("threads participating", f"{OURS['threads_tot']:,}", f"{LLAMA['threads_tot']:,}",
         f"{LLAMA['threads_tot']/OURS['threads_tot']:.1f}x"),
        ("wavefronts", f"{OURS['wf']}", f"{LLAMA['wf']}", f"{LLAMA['wf']/OURS['wf']:.1f}x"),
        ("wavefronts per SIMD (of 16)", f"{OURS['wf']/SIMD:.2f}", f"{LLAMA['wf']/SIMD:.2f}",
         f"{LLAMA['wf']/OURS['wf']:.1f}x"),
        ("state elements per thread", f"{OURS['el_thread']}", f"{LLAMA['el_thread']}",
         f"{LLAMA['el_thread']/OURS['el_thread']:.3f}x"),
        ("slots per token per wavefront", f"{OURS['slots_tok_wf']:,}", f"{LLAMA['slots_tok_wf']:,}",
         f"{LLAMA['slots_tok_wf']/OURS['slots_tok_wf']:.3f}x"),
        ("slots per token, whole grid", f"{b['ours']['slots_tok']:,.0f}", f"{b['llama']['slots_tok']:,.0f}",
         f"{b['llama']['slots_tok']/b['ours']['slots_tok']:.2f}x"),
        ("global state accesses per token", f"{3*STATE_EL:,}", f"{2*STATE_EL/SPAN:,.0f}",
         f"{(2*STATE_EL/SPAN)/(3*STATE_EL):.4f}x"),
        ("bytes demanded at the LSU per token", f"{b['ours']['MB_tok']:.2f} MB",
         f"{b['llama']['MB_tok']*1000:.0f} KB",
         f"{b['llama']['MB_tok']/b['ours']['MB_tok']:.3f}x"),
        ("  of which cannot come from L1 (unique)", f"{b['ours']['MB_tok_unique']:.2f} MB",
         f"{b['llama']['MB_tok_unique']*1000:.0f} KB",
         f"{b['llama']['MB_tok_unique']/b['ours']['MB_tok_unique']:.4f}x"),
        ("demanded GB/s at the LSU", f"{b['ours']['GBs']:,.0f}", f"{b['llama']['GBs']:,.0f}", ""),
        ("demanded GB/s, unique bytes only", f"{b['ours']['GBs_unique']:,.0f}",
         f"{b['llama']['GBs_unique']:,.0f}", ""),
        ("as % of the 918 GB/s cache roof (LSU)", f"{100*b['ours']['GBs']/BW_CACHE_G:,.0f} %",
         f"{100*b['llama']['GBs']/BW_CACHE_G:,.0f} %", ""),
        ("as % of the 918 GB/s cache roof (unique)", f"{100*b['ours']['GBs_unique']/BW_CACHE_G:,.0f} %",
         f"{100*b['llama']['GBs_unique']/BW_CACHE_G:.1f} %", ""),
        ("as % of the 237 GB/s DRAM roof (unique)", f"{100*b['ours']['GBs_unique']/BW_DRAM_G:,.0f} %",
         f"{100*b['llama']['GBs_unique']/BW_DRAM_G:.1f} %", ""),
        ("achieved G slots/s", f"{b['ours']['Gslots']:,.1f}", f"{b['llama']['Gslots']:,.1f}",
         f"{b['llama']['Gslots']/b['ours']['Gslots']:.2f}x"),
        ("as % of the 231.5 G-slot issue roof", f"{b['ours']['pct_issue']:.1f} %",
         f"{b['llama']['pct_issue']:.1f} %", ""),
        ("achieved TFLOP/s (scalar f32)", f"{b['ours']['TFLOPS']:.2f}", f"{b['llama']['TFLOPS']:.2f}", ""),
        ("as % of the 7.30 TFLOP/s scalar fma roof", f"{b['ours']['pct_fma']:.1f} %",
         f"{b['llama']['pct_fma']:.1f} %", ""),
        ("traced ms/pass (S38)", f"{OURS['ms']}", f"{LLAMA['ms']}", f"{OURS['ms']/LLAMA['ms']:.2f}x"),
    ]
    print(f"{'quantity':>42}  {'ours':>16}  {'llama':>16}  ratio")
    for r in rows:
        print(f"{r[0]:>42}  {r[1]:>16}  {r[2]:>16}  {r[3]}")
    print()
    # What would OUR instruction count cost at llama's issue efficiency, and vice versa?
    ours_slots_pass = b["ours"]["slots_tok"] * TOK_LAYERS
    print(f"our slots for the pass: {ours_slots_pass/1e9:.2f} G -> at llama's "
          f"{b['llama']['pct_issue']:.0f} % issue efficiency = "
          f"{ours_slots_pass/(ISSUE_ROOF*b['llama']['pct_issue']/100)*1e3:.1f} ms "
          f"(vs {OURS['ms']} ms now); at 50 % = {ours_slots_pass/(ISSUE_ROOF*0.5)*1e3:.1f} ms")
    ll_slots_pass = b["llama"]["slots_tok"] * TOK_LAYERS
    print(f"a llama-shaped grid (512 wf) carrying ~213 slots/token costs {ll_slots_pass/1e9:.2f} G slots;"
          f" at 74 % of roof = {ll_slots_pass/(ISSUE_ROOF*0.74)*1e3:.1f} ms;"
          f" at 40 % = {ll_slots_pass/(ISSUE_ROOF*0.40)*1e3:.1f} ms;"
          f" at 25 % = {ll_slots_pass/(ISSUE_ROOF*0.25)*1e3:.1f} ms")
    print(f"state arithmetic floor: {STATE_EL*4*TOK_LAYERS/SCALAR_MAC_S*1e3:.1f} ms at the 7.30 TFLOP/s roof")


if __name__ == "__main__":
    main()
