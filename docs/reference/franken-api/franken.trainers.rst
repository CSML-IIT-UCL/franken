..
  module.rst

franken.trainers
================

.. automodule:: franken.trainers

    .. rubric:: Classes

    .. autosummary::
        :toctree:
        :nosignatures:
        :template: autosummary/class.rst

            RandomFeaturesTrainer
            BaseTrainer


    .. autoclass:: BaseTrainer
        :members:
        :no-inherited-members:

            .. rubric:: Methods

            .. autosummary::
                :nosignatures:

                    ~BaseTrainer.fit
                    ~BaseTrainer.evaluate
                    ~BaseTrainer.get_statistics
                    ~BaseTrainer.serialize_logs
                    ~BaseTrainer.serialize_best_model

    .. autoclass:: RandomFeaturesTrainer
        :members:

            .. rubric:: Methods

            .. autosummary::
                :nosignatures:

                    ~RandomFeaturesTrainer.fit
                    ~RandomFeaturesTrainer.evaluate
