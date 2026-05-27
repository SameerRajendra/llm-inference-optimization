import asyncio
import time
import random
import argparse
import aiohttp
import numpy as np
from typing import List, Dict

async def send_request(session: aiohttp.ClientSession, url: str, payload: Dict, req_id: int) -> Dict:
    """Asynchronously streams generated tokens from the endpoint to profile TTFT and ITL metrics."""
    ttft = 0.0
    total_latency = 0.0
    token_timestamps = []
    start_time = time.perf_counter()
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return {"req_id": req_id, "status": "FAIL", "error": f"HTTP {response.status}"}
                
            # Stream chunked chunks from the running serving runtime
            async for chunk in response.content:
                current_time = time.perf_counter()
                if not token_timestamps:
                    # Capture exact Time To First Token (TTFT) at the first arriving frame
                    ttft = current_time - start_time
                token_timestamps.append(current_time)
                
            end_time = time.perf_counter()
            total_latency = end_time - start_time
            num_tokens = len(token_timestamps)
            
            # Extract Inter-Token Latency (ITL) distribution metrics
            itl_deltas = np.diff(token_timestamps) if num_tokens > 1 else [0.0]
            avg_itl = np.mean(itl_deltas) if len(itl_deltas) > 0 else 0.0
            
            return {
                "req_id": req_id,
                "status": "SUCCESS",
                "ttft": ttft,
                "total_latency": total_latency,
                "num_tokens": num_tokens,
                "avg_itl": avg_itl,
                "throughput": num_tokens / total_latency if total_latency > 0 else 0.0
            }
    except Exception as e:
        return {"req_id": req_id, "status": "EXCEPT", "error": str(e)}

async def main(args):
    url = f"http://{args.host}:{args.port}/v1/completions"
    
    # Complex prompt structures to verify scheduling delays under non-homogenous input states
    base_prompts = [
        "Explain the microarchitectural variations of Hopper vs Blackwell SM layouts.",
        "Write a highly optimized CUDA matrix multiplication kernel using shared memory swizzled tiling.",
        "What are the primary indicators of warp stalls and register spills on an H100?",
        "Deconstruct the mathematical formulation and memory traffic savings of Grouped Query Attention."
    ]
    
    tasks = []
    req_id = 0
    
    async with aiohttp.ClientSession() as session:
        print(f"[*] Initializing Asynchronous Poisson Traffic Engine (Target: {args.qps} QPS)...")
        start_engine = time.perf_counter()
        
        while req_id < args.total_requests:
            req_id += 1
            prompt = random.choice(base_prompts)
            
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "stream": True,  # Required to separate prefill (TTFT) from decode (ITL) mechanics
                "temperature": 0.0
            }
            
            task = asyncio.create_task(send_request(session, url, payload, req_id))
            tasks.append(task)
            
            # Enforce Poisson process inter-arrival distribution: -ln(1 - U) / lambda
            interval = -np.log(1.0 - random.random()) / args.qps
            await asyncio.sleep(interval)
            
        print("[*] All injection request channels enqueued. Synchronizing task matrix...")
        results = await asyncio.gather(*tasks)
        
    successes = [r for r in results if r["status"] == "SUCCESS"]
    print(f"\n=== PORFOLIO TELEMETRY ANALYSIS (Success Rate: {len(successes)}/{args.total_requests}) ===")
    if successes:
        avg_ttft = np.mean([r["ttft"] for r in successes]) * 1000
        p99_ttft = np.percentile([r["ttft"] for r in successes], 99) * 1000
        avg_itl = np.mean([r["avg_itl"] for r in successes]) * 1000
        p99_itl = np.percentile([r["avg_itl"] for r in successes], 99) * 1000
        total_tokens = sum([r["num_tokens"] for r in successes])
        total_wall_time = time.perf_counter() - start_engine
        
        print(f"Average TTFT:         {avg_ttft:.2f} ms")
        print(f"P99 TTFT:             {p99_ttft:.2f} ms  <-- Isolates Scheduler Execution Stalls")
        print(f"Average ITL:          {avg_itl:.2f} ms")
        print(f"P99 ITL:              {p99_itl:.2f} ms  <-- Measures Continuous Batching Thresholds")
        print(f"Aggregate Generation: {total_tokens / total_wall_time:.2f} tokens/sec")
    else:
        print("[!] Ingestion streams returned zero active responses. Verify server container process health.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous Poisson Load Injector Suite")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True, help="Hugging Face or model mapping identifier")
    parser.add_argument("--qps", type=float, default=2.5, help="Poisson process lambda frequency density")
    parser.add_argument("--total-requests", type=int, default=60, help="Total execution request volume")
    parser.add_argument("--max-tokens", type=int, default=128, help="Token budget constraint boundary")
    args = parser.parse_args()
    
    asyncio.run(main(args))