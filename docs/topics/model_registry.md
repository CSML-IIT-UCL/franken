(model-registry)=
# Backbones Registry

The available pre-trained GNNs can be listed by running `franken.backbones list`.
As of today, the available models are:

```
--------------------------------AVAILABLE MODELS--------------------------------
* MACE
mace_mp/small
mace_mp/medium
mace_mp/large
mace_mp/small-0b
mace_mp/medium-0b
mace_mp/small-0b2
mace_mp/medium-0b2
mace_mp/large-0b2
mace_mp/medium-0b3
mace_mpa/medium-0
mace_omat/small-0
mace_omat/medium-0
mace_matpes/pbe-0
mace_matpes/r2scan-0
mace_mh/0
mace_mh/1
mace_omol/0_1024
mace_omol/0_4M
mace_off/small
mace_off/medium
mace_off/medium24
mace_off/large
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