from __future__ import annotations
import pynvml

def get_gpu_specs():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name   = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlShutdown()

    # FIX #1: You're on an H200, not H100 — and your kernel uses float32, not BF16.
    # H200 SXM specs:
    #   FP32 CUDA core (no tensor):  33.5  TFLOPS  ← what your kernel actually uses
    #   FP32 Tensor Core (TF32):     989.0 TFLOPS  ← only if using wmma/tensor cores
    #   BF16 Tensor Core:            1,979 TFLOPS  ← irrelevant for float32 kernels
    #   HBM3e bandwidth:             4.8   TB/s    ← H200 upgrade over H100's 3.35

    # FIX #2: Wrong specs were hardcoded (H100 HBM3 @ 3.35 TB/s, not H200 HBM3e @ 4.8 TB/s)
    # H100 SXM5:  989 TFLOPS BF16,  3.35 TB/s HBM3   → ridge = 295 FLOPs/byte
    # H200 SXM:   989 TFLOPS BF16,  4.8  TB/s HBM3e  → ridge = 206 FLOPs/byte
    peak_fp32_cuda    =   33.5   # TFLOPS — your kernel's actual ceiling
    peak_fp32_tensor  =  989.0   # TFLOPS — only reachable with TF32 tensor cores
    peak_bf16_tensor  = 1979.0   # TFLOPS — only with BF16 tensor cores
    memory_bandwidth  =    4.8   # TB/s   — H200 SXM HBM3e

    # FIX #3: Ridge point units were wrong.
    # ridge = peak_compute (FLOP/s) / peak_bandwidth (bytes/s)
    # Both must be in the SAME base units before dividing.
    # OLD (WRONG): (989.0 * 1000) / (3.35 * 1000)  — the *1000 cancels, giving TFLOPS/TB/s
    #              which happens to equal FLOPs/byte numerically, but is misleading
    #              and breaks if you ever use mismatched prefixes.
    # CORRECT: convert both to base units explicitly.
    ridge_fp32_cuda   = (peak_fp32_cuda   * 1e12) / (memory_bandwidth * 1e12)  # FLOPs/byte
    ridge_fp32_tensor = (peak_fp32_tensor * 1e12) / (memory_bandwidth * 1e12)
    ridge_bf16_tensor = (peak_bf16_tensor * 1e12) / (memory_bandwidth * 1e12)

    print(f"\n--- Hardware Profile: {name} ---")
    print(f"  Memory Bandwidth  : {memory_bandwidth} TB/s (HBM3e)")
    print(f"")
    print(f"  FP32 CUDA cores   : {peak_fp32_cuda} TFLOPS  → ridge = {ridge_fp32_cuda:.1f} FLOPs/byte")
    print(f"  FP32 Tensor (TF32): {peak_fp32_tensor} TFLOPS  → ridge = {ridge_fp32_tensor:.1f} FLOPs/byte")
    print(f"  BF16 Tensor       : {peak_bf16_tensor} TFLOPS → ridge = {ridge_bf16_tensor:.1f} FLOPs/byte")
    print(f"")
    print(f"  ★ Your kernel uses FP32 CUDA cores → ridge point is {ridge_fp32_cuda:.1f} FLOPs/byte")
    print(f"    (To reach BF16 ridge you need tensor cores + dtype conversion)")
    print("-" * 50)

    return {
        "name":             name,
        "bandwidth_TBs":    memory_bandwidth,
        "peak_fp32_cuda":   peak_fp32_cuda,
        "peak_fp32_tensor": peak_fp32_tensor,
        "peak_bf16_tensor": peak_bf16_tensor,
        "ridge_fp32_cuda":  ridge_fp32_cuda,
        "ridge_fp32_tensor":ridge_fp32_tensor,
        "ridge_bf16_tensor":ridge_bf16_tensor,
    }

if __name__ == "__main__":
    specs = get_gpu_specs()