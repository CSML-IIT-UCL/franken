"""
Use the list of released MACE models to populate the model registry.
See : https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py
"""

import os
import json

# retrieve urls from mace
from mace.calculators.foundations_models import mace_mp_urls

# add development ones
mace_mp_urls.update({
    "mh-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mh-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    "mace_omol_0_1024": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model",
    "mace_omol_0_4M": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/mace-omol-0-extra-large-4M.model"
})

# update registry
with open("registry.json", "r") as f:
    model_registry = json.load(f)

for mace_id, mace_url in mace_mp_urls.items():
    print(mace_id)
    model_registry[f"mace/{mace_id}"] = {
        "remote": mace_url,
        "local": "mace/" + os.path.basename(mace_url),
        "implemented": True,
        "kind": "mace"
    }

with open("registry.json", "w") as f:
    json.dump(model_registry, f, indent=2)

    