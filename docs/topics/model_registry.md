(model-registry)=
# Backbones Registry

The available pre-trained GNNs can be listed by running `franken.backbones list`.
As of today, the available models are:

```
--------------------------------AVAILABLE MODELS--------------------------------
* MACE
mace/small
mace/medium
mace/large
mace/small-0b
mace/medium-0b
mace/small-0b2
mace/medium-0b2
mace/large-0b2
mace/medium-0b3
mace/medium-mpa-0
mace/small-omat-0
mace/medium-omat-0
mace/mace-matpes-pbe-0
mace/mace-matpes-r2scan-0
mace/mh-0
mace/mh-1
mace/mace_omol_0_1024
mace/mace_omol_0_4M
mace/off-small
mace/off-medium
mace/off24-medium
mace/off-large
* SEVENN
SevenNet0/11July2024
* FAIRCHEM
SchNet/S2EF-OC20-200k
SchNet/S2EF-OC20-2M
SchNet/S2EF-OC20-20M
SchNet/S2EF-OC20-All
--------------------------------------------------------------------------------
```

Models can also be directly downloaded by copying the backbone-ID from the command above into the `download` command

```bash
   franken.backbones download <gnn_backbone_id>
```

Check the command-line help (e.g. `franken.backbones download --help`) for more information.