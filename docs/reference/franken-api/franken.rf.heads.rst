franken.rf.heads
================
This module contains random feature implementations for different kernels

Base Class
----------
.. autosummary::
    :toctree: stubs
    :template: class.rst
    :nosignatures:

    franken.rf.heads.RandomFeaturesHead


Gaussian kernel
---------------
Approximations to the classical Gaussian (or RBF) kernel

.. autosummary::
    :toctree: stubs
    :template: class.rst
    :nosignatures:

    franken.rf.heads.OrthogonalRFF
    franken.rf.heads.MultiScaleOrthogonalRFF
    franken.rf.heads.BiasedOrthogonalRFF

Other kernels
-------------

.. autosummary::
    :toctree: stubs
    :template: class.rst
    :nosignatures:

    franken.rf.heads.Linear
    franken.rf.heads.RandomFeaturesHead
    franken.rf.heads.TensorSketch

Helper Functions
----------------

.. autosummary::
    :toctree: stubs
    :template: func.rst
    :nosignatures:

    franken.rf.heads.initialize_rf
