#include <torch/extension.h>
#include <vector>

void launch_fused_gqa(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int seq_len, int num_q_heads, int num_kv_heads, float scale);

torch::Tensor fused_gqa(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {

    // FIX #1: Input validation — catch shape/device errors early
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
        "All tensors must be on CUDA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
        "All tensors must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32,
        "Only float32 supported");

    // Expected tensor layouts:
    //   Q: [batch_size, num_q_heads,  head_dim]
    //   K: [batch_size, seq_len,      num_kv_heads, head_dim]
    //   V: [batch_size, seq_len,      num_kv_heads, head_dim]
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D: [batch, num_q_heads, head_dim]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D: [batch, seq_len, num_kv_heads, head_dim]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D: [batch, seq_len, num_kv_heads, head_dim]");

    // FIX #2: Correct dimension extraction matching the kernel's layout
    // OLD (WRONG): int seq_len     = K.size(0);  // was treating K as 3D [seq, heads, dim]
    //              int num_q_heads = Q.size(1);   // accidentally correct for Q
    //              int num_kv_heads= K.size(1);   // wrong: this was seq_len, not num_kv_heads
    int batch_size   = Q.size(0);
    int num_q_heads  = Q.size(1);
    int seq_len      = K.size(1);   // K is [batch, seq_len, num_kv_heads, head_dim]
    int num_kv_heads = K.size(2);   // FIX: was K.size(1), which grabbed seq_len by mistake

    // FIX #3: Validate GQA group size is a clean multiple
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
        "num_q_heads (", num_q_heads, ") must be divisible by num_kv_heads (", num_kv_heads, ")");

    // FIX #4: Output shape must include batch dim — same as Q
    // OLD (WRONG): auto O = torch::empty_like(Q);
    // empty_like(Q) works here because Q is [batch, num_q_heads, head_dim],
    // which IS the correct output shape — but only because Q has the batch dim.
    // Being explicit is safer and documents intent.
    auto O = torch::empty({batch_size, num_q_heads, Q.size(2)},
                          Q.options());

    launch_fused_gqa(Q, K, V, O, seq_len, num_q_heads, num_kv_heads, scale);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gqa", &fused_gqa,
          "Fused GQA Attention (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"));
}