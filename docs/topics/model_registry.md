(model-registry)=
# Backbones Registry

The available pre-trained GNNs can be listed by running `franken.backbones list`.
As of today, the available models are:

```
                               DOWNLOADED MODELS
--------------------(/path/to/.franken/gnn_checkpoints)--------------------
MACE-L0 (MACE)
--------------------------------AVAILABLE MODELS--------------------------------
SevenNet0 (sevenn)
MACE-L1 (MACE)
MACE-L2 (MACE)
MACE-OFF-small (MACE)
MACE-OFF-medium (MACE)
MACE-OFF-large (MACE)
SchNet-S2EF-OC20-200k (fairchem)
SchNet-S2EF-OC20-2M (fairchem)
SchNet-S2EF-OC20-20M (fairchem)
SchNet-S2EF-OC20-All (fairchem)
--------------------------------------------------------------------------------
```

Models can also be directly downloaded by copying the backbone-ID from the command above into the `download` command

```bash
   franken.backbones download <gnn_backbone_id>
```

Check the command-line help (e.g. `franken.backbones download --help`) for more information.