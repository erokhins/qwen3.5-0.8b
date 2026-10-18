# First-pass prefill ceiling: FLOPs/token from the model spec against the measured matrix roof.
# All inputs cited in the printed table. This is a desk bound, not a measurement.
h, L, attn_layers, gdn_layers = 1024, 24, 6, 18
vocab, tied = 248320, 485 * 1024 * 1024 / 2          # tied lm_head/embedding bytes -> params
body_params = 1.5457e9 / 2 - tied                    # non-embedding params

# exact matmul N x K from the engine's own shape table (gemm_algo_sweep results.md §3a)
shapes = {'q_proj(4096x1024)': (4096, 1024, attn_layers), 'k+v(512x1024)x2': (512, 1024, attn_layers * 2),
          'o/out(1024x2048)': (1024, 2048, attn_layers + gdn_layers),
          'in_proj_qkv(6144x1024)': (6144, 1024, gdn_layers), 'in_proj_z(2048x1024)': (2048, 1024, gdn_layers),
          'gate+up(3584x1024)x2': (3584, 1024, L * 2), 'down(1024x3584)': (1024, 3584, L)}
f_body = sum(2 * n * k * c for n, k, c in shapes.values())
f_head = 2 * tied                                             # lm_head applied per prefill row
n_ctx = 4096
f_attn = 4 * (n_ctx / 2) * 256 * 8 * attn_layers              # causal-average QK^T + PV per token
print(f'body matmuls (from shapes)   {f_body/1e9:.3f} GFLOP/token   params implied {f_body/2/1e6:.0f}M')
print(f'lm_head per row              {f_head/1e9:.3f} GFLOP/token')
print(f'full attention @4096 ctx     {f_attn/1e9:.3f} GFLOP/token (causal average)')
tot = f_body + f_head + f_attn
print(f'TOTAL                        {tot/1e9:.3f} GFLOP/token   (without lm_head: {(f_body+f_attn)/1e9:.3f})')
print(f'cross-check vs record: doc says whole prefill achieved 1.55 TFLOP/s at 967 tok/s -> {1.55e12/967/1e9:.2f} GFLOP/token')
rate = 4096 / 0.9197                                          # §17 median wall for a 4096-token prefill
print(f'\nengine today: {rate:.0f} tok/s -> aggregate {rate*tot/1e12:.2f} TFLOP/s demanded')
for name, roof in [('measured mix rate (M>=512) low', 22.3e12), ('measured mix rate high', 23.3e12),
                   ('best measured single shape', 37.7e12), ('roof if lm_head only on last row', 23.0e12)]:
    tok = roof / (tot if 'lm_head only' not in name else (f_body + f_attn))
    print(f'  ceiling at {name:36s} = {tok:7.0f} tok/s   we are at {100*rate/tok:5.1f}%')
bw = 237e9
print(f'\nweight-traffic bound for a 512-row wave: {1.5457e9/bw*1000:.1f} ms per wave -> '
      f'{4096/(8*1.5457e9/bw):.0f} tok/s (prefill reads weights once per 512 tokens)')
