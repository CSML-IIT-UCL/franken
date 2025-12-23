"""
Script to automatically generate the MACE part of the model registry using the list of released MACE models.
Some urls are available also inside the mace package: https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py
"""

import os
import json

# retrieve mace_mp models from mace
# from mace.calculators.foundations_models import mace_mp_urls

mace_urls = {
    "mace_mp/small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "mace_mp/medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "mace_mp/large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
    "mace_mp/small-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
    "mace_mp/medium-0b": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model",
    "mace_mp/small-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model",
    "mace_mp/medium-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
    "mace_mp/large-0b2": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
    "mace_mp/medium-0b3": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
    "mace_mpa/medium-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "mace_omat/small-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model",
    "mace_omat/medium-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "mace_matpes/pbe-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "mace_matpes/r2scan-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    "mace_mh/0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mace_mh/1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    "mace_omol/0_1024": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model",
    "mace_omol/0_4M": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/mace-omol-0-extra-large-4M.model",
    "mace_off/small": "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true",
    "mace_off/medium": "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true",
    "mace_off/medium24": "https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true",
    "mace_off/large": "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model?raw=true",
}

if __name__ == "__main__":
    # update registry
    with open("registry.json", "r") as f:
        model_registry = json.load(f)

    for mace_id, mace_url in mace_urls.items():
        print(mace_id)
        model_registry[f"{mace_id}"] = {
            "remote": mace_url,
            "local": "mace/" + os.path.basename(mace_url),
            "implemented": True,
            "kind": "mace",
        }

    with open("registry.json", "w") as f:
        json.dump(model_registry, f, indent=2)
