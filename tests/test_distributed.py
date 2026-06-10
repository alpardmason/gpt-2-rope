from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel

from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            return int(server.getsockname()[1])
    except PermissionError:
        pytest.skip("environment forbids local sockets required by torch.distributed")


def _ddp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(5)
        model = DistributedDataParallel(
            GPT(
                ModelConfig(
                    vocab_size=32,
                    context_length=8,
                    d_model=16,
                    num_layers=1,
                    num_heads=2,
                    num_kv_heads=1,
                    dropout=0.0,
                )
            )
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        output = model(tokens, labels=tokens, use_cache=False)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()
        checksum = torch.stack([parameter.detach().sum() for parameter in model.parameters()]).sum()
        gathered = [torch.zeros_like(checksum) for _ in range(world_size)]
        distributed.all_gather(gathered, checksum)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0])
    finally:
        distributed.destroy_process_group()


@pytest.mark.distributed
def test_two_process_cpu_gloo_step() -> None:
    multiprocessing.spawn(_ddp_worker, args=(2, _free_port()), nprocs=2, join=True)
