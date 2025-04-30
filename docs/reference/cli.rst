.. _cli_reference:

Franken CLI Reference
=====================


Autotune
--------

.. code-block:: bash

    usage: franken.autotune [-h] [OPTIONS] {backbone:mace,backbone:fairchem,backbone:sevenn}


.. argparse::
    :module: franken.autotune.script
    :func: get_parser_fn
    :prog: franken.autotune
    :nodefault:

    subcommands : @skip
        skip subcommands. This is far from ideal since we'd like subcommands to show up, but current output is worse than nothing.


Backbones
---------


.. argparse::
    :module: franken.backbones.cli
    :func: get_parser_fn
    :prog: franken.backbones


Create LAMMPS model
-------------------

.. argparse::
    :module: franken.calculators.lammps_calc
    :func: get_parser_fn
    :prog: franken.create_lammps_model
