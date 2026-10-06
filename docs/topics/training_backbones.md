(model-registry)=
# GNN backbones

In franken, the input descriptors for learning the potential energy surface are extracted from a pre-trained graph neural network (GNN).
This GNN can be either one of the many general-purpose models or a custom one (at the present, we support MACE and PET architectures).

### General-purpose models

The available pre-trained GNNs can be listed by running `franken.backbones list`:

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

* PET 
PET_MAD/xs_1.5
PET_MAD/s_1.5
PET_OMat/xs_1.0
PET_OMat/s_1.0
PET_OMat/m_1.0
PET_OMat/l_1.0
PET_OMat/xl_1.0

--------------------------------------------------------------------------------
```

When requested for the first time, the model will be downloaded into the franken cache directory (by default: `$HOME/.franken`)  

Notes:
* For general applications, we recommend starting with the 
`mace_mh/0` model (or the more expressive - but more expensive - variant `mace_mh/1`).

* It is best to choose a backbone whose pretraining domain contains chemical
environments and elements similar to the target system. As a general guideline, the
model families cover the following domains:

| Model family | Pretraining domain and intended use |
| --- | --- |
| **MACE-MP** | Predominantly inorganic, periodic materials derived from Materials Project data. |
| **MACE-MPA, MACE-OMat, MACE-MatPES, and PET-OMat** | Broad inorganic materials represented by large, diverse datasets. |
| **MACE-MH** | Multiple chemical domains.  This is the recommended general-purpose starting point. |
| **MACE-OMol and MACE-OFF** | Molecules and organic chemistry. |
| **PET-MAD** | Chemically diverse atomistic systems; the broader-domain PET choice. |

* Within one training domain and model generation, larger size variants generally provide
a more expressive representation and can yield a more accurate potential, but increase
the training and inference cost. 

### Custom models

Franken can also use a compatible MACE or PET model that is not in the registry. Pass
the checkpoint path instead of a registered ID through `path_or_id`:

```python
from franken.config import MaceBackboneConfig, PETBackboneConfig

mace_config = MaceBackboneConfig(path_or_id="/path/to/mace.model")
pet_config = PETBackboneConfig(path_or_id="/path/to/pet.ckpt")
```

The corresponding CLI forms are 
- `--backbone=mace --mace.path-or-id /path/to/mace.model`
- `--backbone=pet --pet.path-or-id /path/to/pet.ckpt`
