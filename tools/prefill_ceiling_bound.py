"""Absolute ceilings for this engine, from the model spec against measured device rates.

Desk bound only — CPU arithmetic on cited numbers, no device work. Re-run after any change to the
family table or the measured rates. Inputs:
  shapes/dims   models/Qwen3.5-0.8B/config.json  +  prototype/gemm_algo_sweep/results.md §3a
  weights       specification/01 §01.7  (1.5457 GB bf16 text path, 485 MiB tied lm_head/embedding)
  roof          devices/strix-halo-compute.md "Second correction" (16-bit C): 33.1-37.7 TFLOP/s single
                shape, 22.3-23.3 TFLOP/s on the 24-layer wave mix at M>=512, window prefill-Wclean
  bandwidth     devices/strix-halo-memory-bandwidth.md (237 GB/s ceiling, 164-168 GB/s plateau)
  family table  a `--time-waves` run: tmp/gcc/rr_r3.txt (920.18 ms wave at 4096 lanes)
"""
import re, sys

CFG = dict(hidden=1024, layers=24, attn_layers=6, gdn_layers=18, head_dim=256, attn_heads=8,
           kv_heads=2, inter=3584, vocab=248320)
WEIGHT_GB, TIED_MIB = 1.5457, 485
MIX_LO, MIX_HI, PEAK = 22.3e12, 23.3e12, 37.7e12
BW, PLATEAU = 237e9, 164e9

# ---- FLOPs per prefill token, timed path. lm_head is ONE launch per prefill (prefill.cpp:91 prices
# only row n-1 under the default --prefill-logits last), so it is a per-wave constant, not per token.
shapes = [('q_proj 4096x1024', 4096, 1024, CFG['attn_layers']),
          ('k_proj 512x1024', 512, 1024, CFG['attn_layers']),
          ('v_proj 512x1024', 512, 1024, CFG['attn_layers']),
          ('o_proj/out_proj 1024x2048', 1024, 2048, CFG['layers']),
          ('in_proj_qkv 6144x1024', 6144, 1024, CFG['gdn_layers']),
          ('in_proj_z 2048x1024', 2048, 1024, CFG['gdn_layers']),
          ('gate_proj 3584x1024', 3584, 1024, CFG['layers']),
          ('up_proj 3584x1024', 3584, 1024, CFG['layers']),
          ('down_proj 1024x3584', 1024, 3584, CFG['layers'])]
f_body = sum(2.0 * n * k * c for _, n, k, c in shapes)
n_ctx = 4096
f_attn = 4.0 * (n_ctx / 2.0) * CFG['head_dim'] * CFG['attn_heads'] * CFG['attn_layers']
f_lm_head_wave = 2.0 * CFG['vocab'] * CFG['hidden']
tot = f_body + f_attn
print('== FLOPs per prefill token (timed path, lm_head = last row only)')
for nm, n, k, c in shapes:
    print(f'   {nm:26s} x{c:3d}  {2.0*n*k*c/1e9:.4f} GFLOP/token')
print(f'   body total                 {f_body/1e9:.4f} GFLOP/token')
print(f'   full attention @4096 ctx   {f_attn/1e9:.4f} GFLOP/token  (causal average, 6 layers)')
print(f'   lm_head                    {f_lm_head_wave/1e9:.4f} GFLOP per WAVE (1 launch), '
      f'{f_lm_head_wave/1e9/n_ctx*1000:.4f} MFLOP/token amortised')
print(f'   TOTAL                      {tot/1e9:.4f} GFLOP/token')

# ---- measured wave
LOG = sys.argv[1] if len(sys.argv) > 1 else 'tmp/gcc/rr_r3.txt'
s = open(LOG).read()
wall = float(re.search(r'wall_ms=([\d.]+)', s).group(1))
fams = {f: (float(m), int(l)) for f, m, l in
        re.findall(r'gran=stage family=(\S+)\s+ms=([\d.]+) [^\n]*?launches=(\d+)', s)}
tok = 4096
rate = tok / (wall / 1000.0)
print(f'\n== measured wave ({LOG}): wall {wall:.1f} ms -> {rate:.0f} tok/s')
demanded = rate * tot
print(f'   demanded arithmetic        {demanded/1e12:.2f} TFLOP/s aggregate')
print(f'   ceiling at mix roof {MIX_LO/1e12:.1f}-{MIX_HI/1e12:.1f} TFLOP/s: '
      f'{MIX_LO*1.0/tot:,.0f} - {MIX_HI/tot:,.0f} tok/s -> we are at '
      f'{100*rate*tot/MIX_HI:.1f}-{100*rate*tot/MIX_LO:.1f} %')
print(f'   ceiling at single-shape peak {PEAK/1e12:.1f} TFLOP/s: {PEAK/tot:,.0f} tok/s -> '
      f'{100*rate*tot/PEAK:.1f} %')

# ---- per family: what each one costs vs what its own roof would allow
matmul = [f for f in ('proj', 'mlp_gateup', 'mlp_down') if f in fams]
mm_ms = sum(fams[f][0] for f in matmul)
mm_flop = f_body * tok
print(f'\n== the finding: the GEMM families alone')
print(f'   {matmul} = {mm_ms:.2f} ms carrying {mm_flop/1e12:.3f} TFLOP')
print(f'   achieved                   {mm_flop/(mm_ms/1000.0)/1e12:.2f} TFLOP/s  '
      f'vs mix roof {MIX_LO/1e12:.1f}-{MIX_HI/1e12:.1f} -> '
      f'{100*(mm_flop/(mm_ms/1000.0))/MIX_HI:.0f}-{100*(mm_flop/(mm_ms/1000.0))/MIX_LO:.0f} % of it')
print(f'   i.e. if matmul work were FREE the wave would lose only {mm_ms:.0f} ms of {wall:.0f} ms')
attn_ms = fams['attn'][0]
print(f'\n== attention')
print(f'   {attn_ms:.2f} ms for {f_attn*tok/1e12:.3f} TFLOP = {f_attn*tok/(attn_ms/1000.0)/1e12:.2f} TFLOP/s '
      f'({100*f_attn*tok/(attn_ms/1000.0)/MIX_HI:.1f} % of mix roof) - it is a streaming small-M shape, '
      f'not a GEMM')
rec_ms = fams['gdn_rec'][0]
print(f'\n== gdn_rec: {rec_ms:.2f} ms = {100*rec_ms/wall:.1f} % of wall, no matmul at all')
print(f'   {fams["gdn_rec"][1]:,} launches, {rec_ms*1000/fams["gdn_rec"][1]:.2f} us each; a serial latency-bound')
print(f'   scan. This term is what the matrix roof cannot answer.')
# ---- bytes bounds
print(f'\n== bytes bounds')
print(f'   prefill: a 512-row wave reads the weights once -> {WEIGHT_GB*1e9/BW*1000:.1f} ms/wave '
      f'-> {tok/(8*WEIGHT_GB*1e9/BW):,.0f} tok/s at {BW/1e9:.0f} GB/s: not binding')
dec = 1.0 / 71.3
ceil_dec = BW / (WEIGHT_GB * 1e9)
plat_dec = PLATEAU / (WEIGHT_GB * 1e9)
print(f'   decode: {WEIGHT_GB} GB/token at {BW/1e9:.0f} GB/s -> {1000.0/ceil_dec:.2f} ms/token -> '
      f'{ceil_dec:.1f} tok/s absolute bound; at the {PLATEAU/1e9:.0f} GB/s plateau '
      f'{plat_dec:.1f} tok/s')
print(f'   today 71.3 tok/s = {71.3/ceil_dec*100:.0f} % of the ceiling bound, '
      f'{71.3/plat_dec*100:.0f} % of the plateau bound')
