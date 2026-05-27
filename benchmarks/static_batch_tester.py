import asyncio
import time
import httpx
import numpy as np

API_URL = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

async def send_request(client, prompt_id):
    payload = {
        "model": MODEL,
        "prompt": "Explain the concept of CUDA memory tiling in one paragraph.",
        "max_tokens": 128,
        "stream": False
    }
    
    start_time = time.perf_counter()
    response = await client.post(API_URL, json=payload, timeout=120)
    end_time = time.perf_counter()
    
    if response.status_code == 200:
        data = response.json()
        tokens = data['usage']['completion_tokens']
        latency = end_time - start_time
        return tokens / latency, latency # TPS, Total Latency
    return 0, 0

async def run_benchmark(concurrency):
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        
        tps_list = [r[0] for r in results if r[0] > 0]
        print(f"--- Results for Concurrency: {concurrency} ---")
        print(f"Avg Throughput: {np.mean(tps_list):.2f} tokens/s")
        print(f"P99 Latency: {np.percentile([r[1] for r in results], 99):.4f}s")

if __name__ == "__main__":
    for c in [1, 8, 32]: # Concurrency levels from your plan 
        asyncio.run(run_benchmark(c))