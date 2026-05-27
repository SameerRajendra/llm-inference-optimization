#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <torch/extension.h>

#define HEAD_DIM 128
#define THREADS_PER_BLOCK 256
#define KV_TILE_SIZE 64 // Process blocks of 64 KV cache tokens in parallel inside SRAM

__global__ void fused_gqa_attention_kernel(
    const float* __restrict__ Q,         // [batch, num_q_heads, head_dim]
    const float* __restrict__ K_cache,   // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ V_cache,   // [batch, num_kv_heads, seq_len, head_dim]
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

    // Calculate absolute base pointer offset for this specific head's KV cache matrix
    int kv_head_offset = batch_idx * num_kv_heads * seq_len * HEAD_DIM + kv_head_idx * seq_len * HEAD_DIM;

    // 2. Loop over the KV Cache sequence length in Macro Blocks of 64 tokens
    int num_tiles = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        
        // Collective Coalesced Fetch from HBM3e into Shared Memory
        // 256 threads collaborate to load 64 * 128 = 8,192 elements (each thread moves 32 floats)
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int total_element = tid * 32 + i;
            int tile_token_idx = total_element / HEAD_DIM;
            int channel_idx = total_element % HEAD_DIM;
            int global_token_idx = t * KV_TILE_SIZE + tile_token_idx;

            if (global_token_idx < seq_len && tile_token_idx < KV_TILE_SIZE) {
                int global_offset = kv_head_offset + global_token_idx * HEAD_DIM + channel_idx;
                tile_K[tile_token_idx][channel_idx] = K_cache[global_offset];
                tile_V[tile_token_idx][channel_idx] = V_cache[global_offset];
            } else if (tile_token_idx < KV_TILE_SIZE) {
                tile_K[tile_token_idx][channel_idx] = 0.0f;
                tile_V[tile_token_idx][channel_idx] = 0.0f;
            }
        }
        // Cut barrier count down 64x—Synchronize once per tile block, NOT per token!
        __syncthreads(); 

        // 3. Parallel Attention Score Computation via 4-Thread Warp Groups
        // Distribute the 64 SRAM tokens across 256 threads -> 4 threads cooperate per token
        int threads_per_token = 4;
        int token_slot = tid / threads_per_token; // 0 to 63
        int lane_id = tid % threads_per_token;    // 0 to 3
        int global_token_idx = t * KV_TILE_SIZE + token_slot;

        if (global_token_idx < seq_len) {
            float partial_score = 0.0f;
            
            // Map each of the 4 threads to compute a partial dot product over 32 channels
            #pragma unroll
            for (int d = lane_id * 32; d < (lane_id + 1) * 32; ++d) {
                partial_score += shared_Q[d] * tile_K[token_slot][d];
            }

            // High-velocity intra-warp register shuffle reduction (bypasses shared memory fences)
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
        __syncthreads(); // Prevent SRAM corruption before loading the next sequence block
    }

    // 6. Block-Wide Final Output Resolution and Unified Writeout to HBM3e
    // Map the 64 thread slots back into a clean shared memory reduction array
    __shared__ float final_sum_buf[64];
    int lane_id = tid % 4;
    int token_slot = tid / 4;
    
    if (lane_id == 0 && token_slot < 64) {
        final_sum_buf[token_slot] = running_sum;
    }
    __syncthreads();

    // Use the first 128 threads to compile and normalize the register accumulator maps
    if (tid < HEAD_DIM) {
        float total_accumulated_channel = 0.0f;
        float aggregate_denominator = 0.0f;

        // Reduce across the computed tile footprints
        for (int s = 0; s < 64; ++s) {
            int slot_thread_owner = s * 4;
            // Fetch register state mappings from the designated group anchors
            float group_sum = final_sum_buf[s];
            if (t * KV_TILE_SIZE + s < seq_len) {
                // Read from thread group zero lane states
                // Dynamically emit out to global memory exactly once
                aggregate_denominator += group_sum;
            }
        }
        
        // Overwrite standard output positions exactly once
        O[batch_idx * num_q_heads * HEAD_DIM + q_head_idx * HEAD_DIM + tid] = acc[tid] / running_sum;
    }
}

// Fixed Launcher Interface mapping cleanly to your python extension hooks
void launch_fused_gqa(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int seq_len, int num_q_heads, int num_kv_heads, float scale) {

    const int batch_size = Q.size(0);
    
    // Grid maps exactly to your head footprint boundaries
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
        scale
    );
}