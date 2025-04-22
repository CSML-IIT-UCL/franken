..
  module.rst

franken.calculators
===================

.. automodule:: franken.calculators

    .. rubric:: Classes

    .. autosummary::
        :toctree:
        :nosignatures:

            FrankenCalculator
            LammpsFrankenCalculator


    .. autoclass:: FrankenCalculator
        :members: calculate
        :exclude-members: __new__


    .. autoclass:: LammpsFrankenCalculator
        :members: forward, create_lammps_model
        :exclude-members: __new__
