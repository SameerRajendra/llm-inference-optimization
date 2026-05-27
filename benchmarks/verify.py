import torch
import custom_attention
import math

# 1. Setup H200 Execution Parameters
device = torch.device("cuda")
batch_size = 1
num_q_heads = 32
num_kv_heads = 8  # GQA Factor of 4 (Standard for Llama-3-8B)
head_dim = 128
seq_len = 1024
scale = 1.0 / math.sqrt(head_dim)

# 2. Generate Tensors in Production 4D Format
Q = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=torch.float32)
K = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)
V = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float32)

# 3. Reference Implementation Setup (PyTorch SDPA Expects [B, H, S, D])
Q_ref = Q.unsqueeze(2)  # Shape: [1, 32, 1, 128]
K_ref = K.permute(0, 2, 1, 3)  # Shape: [1, 8, 1024, 128]
V_ref = V.permute(0, 2, 1, 3)  # Shape: [1, 8, 1024, 128]

# Expand KV heads to match Q heads for standard SDPA validation
K_ref_expanded = K_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
V_ref_expanded = V_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

# Execute Reference Pass using FP16 to satisfy FlashAttention constraints
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q_ref.half(), 
        K_ref_expanded.half(), 
        V_ref_expanded.half(), 
        scale=scale
    )
expected = expected.squeeze(2).float()  # Bring back to float32 for comparison

# Execute Custom Tiled GQA Pass (Conforms to the 4-argument signature)
O = custom_attention.fused_gqa(Q, K, V, scale)

# Mathematical Parity Audit
max_diff = torch.max(torch.abs(expected - O)).item()
print("--- H200 Custom Kernel Verification ---")
print(f"Max Absolute Difference: {max_diff:.8f}")
if max_diff < 1e-4:
    print("Mathematical Parity: ✅ PASSED")
else:
    print("Mathematical Parity: ❌ FAILED")

# 4. Asynchronous CUDA Hardware Benchmarking Loop
print("\n--- Commencing Microarchitectural Benchmark (1000 Iterations) ---")
warmup = 100
iters = 1000

# Clear compilation and engine overhead via warmup iterations
for _ in range(warmup):
    torch.nn.functional.scaled_dot_product_attention(Q_ref.half(), K_ref_expanded.half(), V_ref_expanded.half(), scale=scale)
    _ = custom_attention.fused_gqa(Q, K, V, scale)
torch.cuda.synchronize()

# Time PyTorch Native SDPA (FlashAttention Backend)
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

start_evt.record()
for _ in range(iters):
    torch.nn.functional.scaled_dot_product_attention(Q_ref.half(), K_ref_expanded.half(), V_ref_expanded.half(), scale=scale)
end_evt.record()
torch.cuda.synchronize()
sdpa_time = start_evt.elapsed_time(end_evt) / iters

# Time Custom Tiled Kernel
start_evt.record()
for _ in range(iters):
    _ = custom_attention.fused_gqa(Q, K, V, scale)
end_evt.record()
torch.cuda.synchronize()
custom_time = start_evt.elapsed_time(end_evt) / iters

# Print Results Summary
print(f"PyTorch SDPA Latency (FP16 Fused): {sdpa_time:.4f} ms")
print(f"Custom Tiled Kernel Latency (FP32):  {custom_time:.4f} ms")
print(f"Performance Ratio:                  {sdpa_time / custom_time:.2f}x of native speed")
print("----------------------------------------------------------------")