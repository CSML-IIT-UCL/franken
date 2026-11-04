import os
from typing import Container, List, Tuple, Union
import warnings
from socket import gethostname

import torch
import torch.distributed
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import hostlist


def slurm_to_env():
    hostname = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0]
    os.environ["MASTER_ADDR"] = hostname
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "33633")
    try:
        world_size = int(os.environ["SLURM_NTASKS"])
    except KeyError:
        world_size = int(os.environ["SLURM_NTASKS_PER_NODE"]) * int(
            os.environ["SLURM_NNODES"]
        )
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["RANK"] = os.environ["SLURM_PROCID"]


def is_torchrun():
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def is_slurm():
    return "SLURM_PROCID" in os.environ


def init(distributed: bool) -> int:
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training is only supported on CUDA")
        if is_torchrun():
            pass
        elif is_slurm():
            slurm_to_env()
        else:
            warnings.warn(
                "Cannot initialize distributed training. "
                "Neither torchrun nor SLURM environment variable were found."
            )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            print(
                f"Distributed initialization at rank {os.environ['RANK']} of {world_size} "
                f"(rank {os.environ['LOCAL_RANK']} on {gethostname()} with "
                f"{torch.cuda.device_count()} GPUs allocated)."
            )
            torch.distributed.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
            )

    device = f"cuda:{get_local_rank()}"
    torch.cuda.set_device(device)
    return get_rank()


def get_local_rank() -> int:
    if torch.distributed.is_initialized():
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def all_reduce(tensor: torch.Tensor, op) -> None:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op)
    return None


def all_sum(tensor: torch.Tensor) -> None:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, torch.distributed.ReduceOp.SUM)
    return None


def all_sum_coalesced(tensors: list[torch.Tensor]) -> None:
    """Sum a list of tensors across all ranks using one collective call."""
    if torch.distributed.is_initialized():
        flat = _flatten_dense_tensors(tensors)
        torch.distributed.all_reduce(flat, torch.distributed.ReduceOp.SUM)
        for t, synced in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
            t.copy_(synced)


def broadcast_obj(obj, src=0):
    if torch.distributed.is_initialized():
        to_broadcast = [obj]
        torch.distributed.broadcast_object_list(to_broadcast, src=src)
        return to_broadcast[0]
    return obj


def all_gather_into_tensor(
    out_size: Union[Tuple, torch.Size], in_tensor: torch.Tensor
) -> torch.Tensor:
    if torch.distributed.is_initialized():
        out_tensor = torch.zeros(
            out_size, dtype=in_tensor.dtype, device=in_tensor.device
        )
        torch.distributed.all_gather_into_tensor(out_tensor, in_tensor)
        return out_tensor
    return in_tensor


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    if torch.distributed.is_initialized():
        shapes = [
            tensor.shape if r == get_rank() else None for r in range(get_world_size())
        ]
        for r in range(get_world_size()):
            shapes[r] = broadcast_obj(shapes[r], src=r)
        tensor_list = [
            (
                tensor
                if r == get_rank()
                else torch.empty(shapes[r], device=tensor.device, dtype=tensor.dtype)  # type: ignore
            )  # type: ignore
            for r in range(get_world_size())
        ]
        torch.distributed.all_gather(tensor_list, tensor)
        return tensor_list
    return [tensor]


def all_gather_object(obj) -> List:
    if torch.distributed.is_initialized():
        output = [None for _ in range(get_world_size())]

        torch.distributed.all_gather_object(output, obj)
        return output
    return [obj]


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def sync_module_states(
    module: torch.nn.Module,
    params_and_buffers_to_ignore: Container[str],
    broadcast_buffers: bool = True,
) -> None:
    """
    Sync ``module``'s parameters and buffers state.

    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
    if not torch.distributed.is_initialized():
        return
    module_states: list[torch.Tensor] = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())

    _sync_params_and_buffers(
        process_group=torch.distributed.group.WORLD,
        module_states=module_states,
        broadcast_bucket_size=250 * 1024 * 1024,
        src=0,
    )


def _sync_params_and_buffers(
    process_group: torch.distributed.ProcessGroup,
    module_states: list[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
) -> None:
    """Synchronize ``module_states`` (list of tensors) across all processes by broadcasting them from rank 0."""
    if len(module_states) > 0:
        torch.distributed._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )
