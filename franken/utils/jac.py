from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torch._functorch.eager_transforms import (
    _construct_standard_basis_for,
    _jvp_with_argnums,
    _slice_argnums,
    error_if_complex,
    safe_unflatten,
)
from torch._functorch.utils import argnums_t
from torch.func import vmap
from torch.utils._pytree import tree_flatten, tree_unflatten

from franken.utils.misc import garbage_collection_cuda, is_cuda_out_of_memory


def jacfwd(
    # drop-in replacement of torch.func.jacfwd accepting the chunk_size argument (as with jacrev)
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
    chunk_size: Optional[int] = None,
):
    def wrapper_fn(*args):
        error_if_complex("jacfwd", args, is_input=True)
        primals = args if argnums is None else _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            output = _jvp_with_argnums(
                func, args, basis, argnums=argnums, has_aux=has_aux
            )
            # output[0] is the output of `func(*args)`
            error_if_complex("jacfwd", output[0], is_input=False)
            if has_aux:
                _, jvp_out, aux = output
                return jvp_out, aux
            _, jvp_out = output
            return jvp_out

        results = vmap(push_jvp, randomness=randomness, chunk_size=chunk_size)(basis)
        if has_aux:
            results, aux = results
            # aux is in the standard basis format, e.g. NxN matrix
            # We need to fetch the first element as original `func` output
            flat_aux, aux_spec = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)

        jac_outs, spec = tree_flatten(results)
        # Most probably below output check can never raise an error
        # as jvp should test the output before
        # assert_non_empty_output(jac_outs, 'jacfwd(f, ...)(*args)')

        jac_outs_ins = tuple(
            tuple(
                safe_unflatten(jac_out_in, -1, primal.shape)
                for primal, jac_out_in in zip(
                    flat_primals,
                    jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1),
                )
            )
            for jac_out in jac_outs
        )
        jac_outs_ins = tuple(
            tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins
        )

        if isinstance(argnums, int):
            jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)
        if has_aux:
            return tree_unflatten(jac_outs_ins, spec), aux
        return tree_unflatten(jac_outs_ins, spec)

    # Dynamo does not support HOP composition if their inner function is
    # annotated with @functools.wraps(...). We circumvent this issue by applying
    # wraps only if we're not tracing with dynamo.
    if not torch.compiler.is_compiling():
        wrapper_fn = wraps(func)(wrapper_fn)

    return wrapper_fn


def tune_jacfwd_chunksize(
    test_sample: Sequence[Union[torch.Tensor, Any]],
    mode: str = "power",
    init_val: int = 32,
    max_trials: int = 25,
    **jac_kwargs,
):
    try:
        # We want to tune this and set it ourselves.
        jac_kwargs.pop("chunk_size")
    except KeyError:
        pass

    # Initially we just double in size until an OOM is encountered
    new_size, _ = _adjust_batch_size(
        test_sample, init_val, value=init_val, **jac_kwargs
    )  # initially set to init_val
    if mode == "power":
        new_size = _run_power_scaling(new_size, max_trials, test_sample, **jac_kwargs)
    else:
        raise ValueError("mode in method `scale_batch_size` can only be `power`")

    garbage_collection_cuda()
    return new_size


def _run_power_scaling(new_size, max_trials, test_sample, **jac_kwargs) -> int:
    """Batch scaling mode where the size is doubled at each iteration until an
    OOM error is encountered."""
    for _ in range(max_trials):
        garbage_collection_cuda()
        try:
            # Try jacfwd
            for _ in range(1):
                jacfwd(**jac_kwargs, chunk_size=new_size)(*test_sample)
            # Double in size
            new_size, changed = _adjust_batch_size(
                test_sample, new_size, factor=2.0, **jac_kwargs
            )
        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_cuda_out_of_memory(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                new_size, _ = _adjust_batch_size(
                    test_sample, new_size, factor=0.5, **jac_kwargs
                )
                break
            else:
                raise  # some other error not memory related
        if not changed:
            # No change in batch size, so we can exit.
            break
    return new_size


def _adjust_batch_size(
    test_sample: Sequence[Union[torch.Tensor, Any]],
    batch_size: int,
    factor: float = 1.0,
    value: Optional[int] = None,
    **jac_kwargs,
) -> Tuple[int, bool]:
    max_batch_size = _get_max_batch_size(test_sample, **jac_kwargs)
    new_size = value if value is not None else int(batch_size * factor)
    new_size = min(new_size, max_batch_size)
    changed = new_size != batch_size
    return new_size, changed


def _get_max_batch_size(
    test_sample: Sequence[Union[torch.Tensor, Any]], **jac_kwargs
) -> int:
    argnums = jac_kwargs.get("argnums", 0)
    if isinstance(argnums, int):
        argnums = [argnums]
    batch_size = 0
    for argnum in argnums:
        arg = test_sample[argnum]
        assert isinstance(arg, torch.Tensor)
        batch_size += arg.numel()
    return batch_size
