import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import custom_attention
import numpy as np

device = torch.device("cuda")

# Llama-3.1-8B GQA Config
num_q_heads  = 32
num_kv_heads = 8
head_dim     = 128
seq_len      = 2048
batch_size   = 1
scale        = 1.0 / (head_dim ** 0.5)

# ---------------------------------------------------------------
# FIX #1: Tensor shapes must match the kernel's expected layout
#   Q: [batch, num_q_heads,  head_dim]
#   K: [batch, seq_len, num_kv_heads, head_dim]   <-- was 3D, missing batch
#   V: [batch, seq_len, num_kv_heads, head_dim]   <-- same
# ---------------------------------------------------------------
Q = torch.randn(batch_size, num_q_heads,  head_dim,               device=device)
K = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,      device=device)
V = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,      device=device)

# ---------------------------------------------------------------
# FIX #2: SDPA reference was using K_exp for both K AND V (typo).
#   scaled_dot_product_attention(Q_ref, K_exp, K_exp)  <- V should be V_exp
#
# FIX #3: SDPA expansion was wrong — repeat_interleave on dim=2 is wrong.
#   K shape after unsqueeze(0): [1, seq_len, num_kv_heads, head_dim]
#   SDPA expects:               [batch, heads, seq_len, head_dim]
#   Correct expansion for GQA reference:
#     1. Permute to [batch, num_kv_heads, seq_len, head_dim]
#     2. repeat_interleave by group_size=4 on dim=1 → [batch, num_q_heads, seq_len, head_dim]
# ---------------------------------------------------------------
group_size = num_q_heads // num_kv_heads  # = 4

# Correct SDPA reference shapes: [batch, heads, seq_len, head_dim]
Q_ref = Q.unsqueeze(2)                              # [1, 32, 1, 128]  (decode: 1 query token)
K_ref = K.permute(0, 2, 1, 3)                      # [1, 8, 2048, 128]
V_ref = V.permute(0, 2, 1, 3)                      # [1, 8, 2048, 128]

# Expand KV heads to match Q heads for SDPA (GQA expansion)
K_ref = K_ref.repeat_interleave(group_size, dim=1) # [1, 32, 2048, 128]
V_ref = V_ref.repeat_interleave(group_size, dim=1) # [1, 32, 2048, 128]


def profile_it(func, name, warmups=10, iterations=100):
    # Warmup
    for _ in range(warmups):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        func()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iterations
    print(f"  {name}: {ms:.4f} ms")
    return ms


# ---------------------------------------------------------------
# FIX #4: Correctness check BEFORE benchmarking.
#   Always verify your kernel matches the reference — if outputs
#   differ, speedup numbers are meaningless.
# ---------------------------------------------------------------
with torch.no_grad():
    ref_out    = torch.nn.functional.scaled_dot_product_attention(Q_ref, K_ref, V_ref)
    # ref_out shape: [1, 32, 1, 128] → squeeze to [1, 32, 128] to match kernel output
    ref_out    = ref_out.squeeze(2)
    custom_out = custom_attention.fused_gqa(Q, K, V, scale)

    max_diff = (ref_out - custom_out).abs().max().item()
    mean_diff = (ref_out - custom_out).abs().mean().item()
    print(f"\n--- Correctness Check ---")
    print(f"  Max  |ref - custom|: {max_diff:.6f}")
    print(f"  Mean |ref - custom|: {mean_diff:.6f}")
    if max_diff < 1e-3:
        print("  ✓ PASS — outputs match within tolerance")
    else:
        print("  ✗ FAIL — outputs diverge, fix kernel before benchmarking!")


# ---------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------
print(f"\n--- H200 Performance Report ---")
pt_time     = profile_it(
    lambda: torch.nn.functional.scaled_dot_product_attention(Q_ref, K_ref, V_ref),
    "PyTorch SDPA"
)
custom_time = profile_it(
    lambda: custom_attention.fused_gqa(Q, K, V, scale),
    "Custom H200 Kernel"
)

print(f"\n  Speedup: {pt_time / custom_time:.2f}x")
print(f"  (seq_len={seq_len}, num_q_heads={num_q_heads}, "
      f"num_kv_heads={num_kv_heads}, head_dim={head_dim})")