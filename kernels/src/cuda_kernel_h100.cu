#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <torch/extension.h>

#define HEAD_DIM 128
#define THREADS_PER_BLOCK 256
#define KV_TILE_SIZE 64 // Process blocks of 64 KV tokens in parallel inside SRAM

__global__ void fused_gqa_attention_kernel(
    const float* __restrict__ Q,         // [batch, num_q_heads, head_dim]
    const float* __restrict__ K_cache,   // [batch, seq_len, num_kv_heads, head_dim]
    const float* __restrict__ V_cache,   // [batch, seq_len, num_kv_heads, head_dim]
    float* __restrict__ O,               // Output [batch, num_q_heads, head_dim]
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    float scale
) {
    int tid = threadIdx.x;
    int q_head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    int group_size = num_q_heads / num_kv_heads; 
    int kv_head_idx = q_head_idx / group_size;

    // Allocate high-speed SRAM Tiles
    __shared__ float shared_Q[HEAD_DIM];
    __shared__ float tile_K[KV_TILE_SIZE][HEAD_DIM];
    __shared__ float tile_V[KV_TILE_SIZE][HEAD_DIM];

    // 1. Coalesced Collaborative Load of the single Query Token into SRAM
    if (tid < HEAD_DIM) {
        shared_Q[tid] = Q[batch_idx * num_q_heads * HEAD_DIM + q_head_idx * HEAD_DIM + tid];
    }

    // Initialize online softmax tracking registers locally per thread
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[HEAD_DIM] = {0.0f};

    // Precompute constant stride dimensions for 4D indexing
    int batch_stride = seq_len * num_kv_heads * HEAD_DIM;
    int token_stride = num_kv_heads * HEAD_DIM;

    // 2. Loop over the KV Cache sequence length in Macro Blocks of 64 tokens
    int num_tiles = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        
        // Collective Coalesced Fetch from HBM3e into Shared Memory
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int total_element = tid * 32 + i;
            int tile_token_idx = total_element / HEAD_DIM;
            int channel_idx = total_element % HEAD_DIM;
            int global_token_idx = t * KV_TILE_SIZE + tile_token_idx;

            if (global_token_idx < seq_len && tile_token_idx < KV_TILE_SIZE) {
                // 4D Stride Mapping: [batch, seq_len, num_kv_heads, head_dim]
                int global_offset = batch_idx * batch_stride + 
                                    global_token_idx * token_stride + 
                                    kv_head_idx * HEAD_DIM + 
                                    channel_idx;
                                    
                tile_K[tile_token_idx][channel_idx] = K_cache[global_offset];
                tile_V[tile_token_idx][channel_idx] = V_cache[global_offset];
            } else if (tile_token_idx < KV_TILE_SIZE) {
                tile_K[tile_token_idx][channel_idx] = 0.0f;
                tile_V[tile_token_idx][channel_idx] = 0.0f;
            }
        }
        __syncthreads(); // Synchronize once per tile block, NOT per token!

        // 3. Parallel Attention Score Computation via 4-Thread Warp Groups
        int threads_per_token = 4;
        int token_slot = tid / threads_per_token; // 0 to 63
        int lane_id = tid % threads_per_token;    // 0 to 3
        int global_token_idx = t * KV_TILE_SIZE + token_slot;

        if (global_token_idx < seq_len) {
            float partial_score = 0.0f;
            
            #pragma unroll
            for (int d = lane_id * 32; d < (lane_id + 1) * 32; ++d) {
                partial_score += shared_Q[d] * tile_K[token_slot][d];
            }

            // High-velocity intra-warp register shuffle reduction
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 1);
            partial_score += __shfl_xor_sync(0xffffffff, partial_score, 2);

            // Lane 0 of the 4-thread group updates the running online softmax allocations
            if (lane_id == 0) {
                float score = partial_score * scale;

                // 4. Mathematical Online Softmax Correction Pass
                float old_max = running_max;
                running_max = fmaxf(running_max, score);
                float exp_score = expf(score - running_max);
                float rescale = expf(old_max - running_max);
                
                running_sum = running_sum * rescale + exp_score;

                // 5. Value Vector Blended Aggregation
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc[d] = acc[d] * rescale + exp_score * tile_V[token_slot][d];
                }
            }
        }
        __syncthreads(); // Prevent SRAM corruption before loading next block
    }

    // 6. Block-Wide Final Output Resolution and Unified Writeout to HBM3e
    __shared__ float smem_max[64];
    __shared__ float smem_sum[64];
    int slot = tid / 4;
    int lane_id = tid % 4;

    if (lane_id == 0) {
        smem_max[slot] = running_max;
    }
    __syncthreads();

    // Warp-level parallel reduction to extract global block maximum
    if (tid < 32) {
        float m = fmaxf(smem_max[tid], smem_max[tid + 32]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
        }
        if (tid == 0) {
            smem_max[0] = m;
        }
    }
    __syncthreads();
    float global_max = smem_max[0];

    if (lane_id == 0) {
        float rescale = expf(running_max - global_max);
        running_sum *= rescale;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            acc[d] *= rescale;
        }
        smem_sum[slot] = running_sum;
    }
    __syncthreads();

    // Warp-level parallel reduction to compute global block denominator
    if (tid < 32) {
        float s_val = smem_sum[tid] + smem_sum[tid + 32];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            s_val += __shfl_down_sync(0xffffffff, s_val, offset);
        }
        if (tid == 0) {
            smem_sum[0] = s_val;
        }
    }
    __syncthreads();
    float global_sum = smem_sum[0];

    // Phase C: Coalesced parallel reduction of accumulator channels
    float* shared_acc = (float*)tile_K; // shape [64][128]
    if (lane_id == 0) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            shared_acc[slot * HEAD_DIM + d] = acc[d];
        }
    }
    __syncthreads();

    if (tid < 256) {
        int channel = tid / 2;
        int part = tid % 2;
        float local_chan_sum = 0.0f;
        int start_slot = part * 32;
        
        #pragma unroll
        for (int s = 0; s < 32; ++s) {
            local_chan_sum += shared_acc[(start_slot + s) * HEAD_DIM + channel];
        }
        
        float total_chan_sum = local_chan_sum + __shfl_xor_sync(0xffffffff, local_chan_sum, 1);
        
        if (part == 0) {
            O[batch_idx * num_q_heads * HEAD_DIM + q_head_idx * HEAD_DIM + channel] = total_chan_sum / global_sum;
        }
    }
}

// Fixed Launcher Interface: Accepts 4 parameters from Python, extracts shapes,
// allocates O inside C++, launches the grid, and returns the output tensor.
torch::Tensor launch_fused_gqa(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {
    const int batch_size = Q.size(0);
    const int num_q_heads = Q.size(1);
    const int head_dim = Q.size(2);
    
    const int seq_len = K.size(1);
    const int num_kv_heads = K.size(2);

    auto O = torch.zeros_like(Q);

    dim3 grid(num_q_heads, batch_size);
    dim3 block(THREADS_PER_BLOCK);

    fused_gqa_attention_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        seq_len,
        num_q_heads,
        num_kv_heads,
        static_cast<float>(scale)
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gqa", &launch_fused_gqa, "Fused GQA Attention Tiled Macro-Kernel");
}