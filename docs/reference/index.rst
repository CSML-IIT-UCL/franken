
Franken API Reference
=====================

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`franken.trainers <franken-api/franken.trainers>`
     - Train franken from atomistic simulation data
   * - :doc:`franken.calculators <franken-api/franken.calculators>`
     - Run molecular dynamics with learned potentials.
   * - :doc:`franken.rf.model <franken-api/franken.rf.model>`
     - Main model class for franken
   * - :doc:`franken.rf.heads <franken-api/franken.rf.heads>`
     - Random feature implementations for different kernels
   * - :doc:`franken.rf.scaler <franken-api/franken.rf.scaler>`
     - Utilities for scaling random features
   * - :doc:`franken.config <franken-api/franken.config>`
     - Configuration data-classes for the whole franken library


.. toctree::
    :maxdepth: 1
    :hidden:

    franken-api/franken.trainers
    franken-api/franken.calculators
    franken-api/franken.rf.model
    franken-api/franken.rf.heads
    franken-api/franken.rf.scaler
    franken-api/franken.config
