Franken Inference Wrappers
==========================

A learned `franken` model can be used directly for simple MD simulations via the :meth:`~franken.calculators.ase_calc.FrankenCalculator` calculator
based on `ASE <https://ase-lib.org/>`_.
To run more complex simulations, for example with `LAMMPS <https://www.lammps.org/>`_, we first need to export the model.
This will both:

 - speed up calculations since exported models are usually JIT compiled,
 - allow to call the model from external programs such as LAMMPS.

We provide two *inference wrappers* which can be saved with the command line utilities described below.


MACE LAMMPS Wrapper
-------------------

.. argparse::
    :module: franken.calculators.mace_inf_wrap
    :func: get_parser_fn
    :prog: franken.wrap_mace_lammps


Metatomic Wrapper
-----------------

.. argparse::
    :module: franken.calculators.metatomic_inf_wrap
    :func: get_parser_fn
    :prog: franken.wrap_metatomic
