from __future__ import annotations
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def run_profile_step():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # A simple large model to force sharding
    model = nn.Sequential(*[nn.Linear(8192, 8192) for _ in range(10)]).to(device)
    model = FSDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(32, 8192).to(device)

    # Nsight Profiler Range
    torch.cuda.nvtx.range_push("FSDP_Step")
    optimizer.zero_grad()
    output = model(data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    torch.cuda.nvtx.range_pop()

    cleanup()

if __name__ == "__main__":
    run_profile_step()