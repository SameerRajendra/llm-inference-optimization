import torch
import custom_attention

device = torch.device("cuda")
num_q_heads  = 32
num_kv_heads = 8
head_dim     = 128
seq_len      = 2048
batch_size   = 1
scale        = 1.0 / (head_dim ** 0.5)

Q = torch.randn(batch_size, num_q_heads,  head_dim,                device=device)
K = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,       device=device)
V = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,       device=device)

# ---------------------------------------------------------------
# Roofline: compute theoretical minimums
# ---------------------------------------------------------------
# Bytes READ:
#   Q: 1  * 32 * 128 * 4 = 16,384 bytes
#   K: 1 * 2048 * 8 * 128 * 4 = 8,388,608 bytes
#   V: 1 * 2048 * 8 * 128 * 4 = 8,388,608 bytes
bytes_read  = (Q.numel() + K.numel() + V.numel()) * 4
bytes_write = Q.numel() * 4  # output O same shape as Q
total_bytes = bytes_read + bytes_write

# FLOPs:
#   Per head: seq_len * HEAD_DIM * 2 (dot product) +
#             seq_len * HEAD_DIM * 2 (weighted sum V)
#   = 2 * 2 * seq_len * HEAD_DIM per Q head
#   Times num_q_heads
flops = 2 * 2 * seq_len * head_dim * num_q_heads * batch_size

# H200 specs
hbm_bandwidth_TBs  = 4.8        # TB/s  (H200 SXM HBM3e peak)
peak_tflops_fp32   = 67.0       # TFLOPS (H200 FP32 tensor core)
peak_tflops_fp32_cuda = 33.5    # TFLOPS (H200 FP32 CUDA core — no tensor)

# Theoretical minimums
min_time_bw_ms    = (total_bytes / (hbm_bandwidth_TBs * 1e12)) * 1e3
min_time_flop_ms  = (flops / (peak_tflops_fp32_cuda * 1e12)) * 1e3
roofline_ms       = max(min_time_bw_ms, min_time_flop_ms)

print(f"\n--- Roofline Analysis ---")
print(f"  Data movement : {total_bytes/1e6:.2f} MB")
print(f"  FLOPs         : {flops/1e6:.2f} MFLOPs")
print(f"  Arithmetic intensity: {flops/total_bytes:.3f} FLOPs/byte")
print(f"  (Roofline ridge point for H200: ~{peak_tflops_fp32_cuda*1e12/(hbm_bandwidth_TBs*1e12):.1f} FLOPs/byte)")
print(f"")
print(f"  Min time (BW-bound)     : {min_time_bw_ms:.4f} ms")
print(f"  Min time (compute-bound): {min_time_flop_ms:.4f} ms")
print(f"  Roofline ceiling        : {roofline_ms:.4f} ms  ← best possible")

# ---------------------------------------------------------------
# Measure actual kernel time (isolated, no Python overhead)
# ---------------------------------------------------------------
def time_kernel(fn, warmups=20, iters=200):
    for _ in range(warmups): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

custom_ms = time_kernel(lambda: custom_attention.fused_gqa(Q, K, V, scale))

# Effective bandwidth and FLOP utilization
eff_bandwidth_TBs = (total_bytes / (custom_ms * 1e-3)) / 1e12
eff_tflops        = (flops / (custom_ms * 1e-3)) / 1e12

print(f"\n--- Kernel Efficiency ---")
print(f"  Actual time         : {custom_ms:.4f} ms")
print(f"  Roofline ceiling    : {roofline_ms:.4f} ms")
print(f"  Efficiency vs roof  : {roofline_ms/custom_ms*100:.1f}%")
print(f"")
print(f"  Effective BW        : {eff_bandwidth_TBs:.3f} TB/s  "
      f"({eff_bandwidth_TBs/hbm_bandwidth_TBs*100:.1f}% of {hbm_bandwidth_TBs} TB/s peak)")
print(f"  Effective TFLOPs    : {eff_tflops:.3f}  "
      f"({eff_tflops/peak_tflops_fp32_cuda*100:.1f}% of {peak_tflops_fp32_cuda} TFLOPS peak)")
print(f"")
# Diagnose
ai = flops / total_bytes
ridge = peak_tflops_fp32_cuda * 1e12 / (hbm_bandwidth_TBs * 1e12)
if ai < ridge:
    print(f"  Kernel is MEMORY-BANDWIDTH BOUND (AI={ai:.2f} < ridge={ridge:.1f})")
    print(f"  → Optimization target: reduce HBM reads (tiling, caching)")
else:
    print(f"  Kernel is COMPUTE BOUND (AI={ai:.2f} > ridge={ridge:.1f})")
    print(f"  → Optimization target: increase FLOP throughput")