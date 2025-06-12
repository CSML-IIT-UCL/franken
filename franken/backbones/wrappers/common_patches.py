import logging

import torch


logger = logging.getLogger("franken")


def patch_e3nn():
    # NOTE:
    #  Patching should occur during training: it is necessary for `jvp` on the MACE model,
    #  but not during inference, when we only use `torch.autograd`. For inference, we may want
    #  to compile the model using `torch.jit` - and the patch interferes with the JIT, so we
    #  must disable it.

    import e3nn.o3._spherical_harmonics

    if hasattr(e3nn.o3._spherical_harmonics._spherical_harmonics, "code"):
        # Then _spherical_harmonics is a scripted function, we need to undo this!
        new_locals = {"Tensor": torch.Tensor}
        exec(e3nn.o3._spherical_harmonics._spherical_harmonics.code, None, new_locals)

        def _spherical_harmonics(
            lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            return new_locals["_spherical_harmonics"](torch.tensor(lmax), x, y, z)

        # Save to allow undoing later
        setattr(
            e3nn.o3._spherical_harmonics,
            "_old_spherical_harmonics",
            e3nn.o3._spherical_harmonics._spherical_harmonics,
        )
        e3nn.o3._spherical_harmonics._spherical_harmonics = _spherical_harmonics

    # 2nd patch for newer e3nn versions (somewhere between 0.5.0 and 0.5.5
    # e3nn jits _spherical_harmonics which the SphericalHarmonics class,
    # making the above patch ineffective)
    try:
        from e3nn import set_optimization_defaults

        set_optimization_defaults(jit_script_fx=False)
    except ImportError:
        pass  # only valid for newer e3nn


def unpatch_e3nn():
    # This is only useful for CI and testing environments.
    # When jit-compiling a franken module (e.g. for LAMMPS), we don't want the patch applied!
    import e3nn.o3._spherical_harmonics

    if hasattr(e3nn.o3._spherical_harmonics, "_old_spherical_harmonics"):
        e3nn.o3._spherical_harmonics._spherical_harmonics = (
            e3nn.o3._spherical_harmonics._old_spherical_harmonics
        )
