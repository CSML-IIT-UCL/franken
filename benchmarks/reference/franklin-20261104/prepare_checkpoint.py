"""Prepare a tiny fitted checkpoint once; never included in inference timings."""
import sys
from pathlib import Path
import numpy as np
import torch
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from franken.config import GaussianRFConfig, MaceBackboneConfig
from franken.data import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
from franken.trainers import RandomFeaturesTrainer

torch.set_num_threads(2)
root = Path(sys.argv[1])
frames = []
rng = np.random.default_rng(17)
for _ in range(2):
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
    atoms.positions += rng.normal(0, .03, atoms.positions.shape)
    atoms.calc = LennardJones(epsilon=.1, sigma=2, rc=5)
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    frames.append(atoms)
write(root/'tiny.xyz', frames)
config = MaceBackboneConfig('mace_mp/medium')
data = FrankenAtomsDataset(root/'tiny.xyz', split='train', gnn_config=config)
model = FrankenPotential(config, GaussianRFConfig(num_random_features=4096, length_scale=2., rng_seed=17), scale_by_Z=False, num_species=1, jac_chunk_size=4).cuda()
trainer = RandomFeaturesTrainer(data.get_dataloader(distributed=False), training_targets=['energy','forces'], l2_penalty=[.01], target_weight={'forces':[.5]}, random_features_normalization=None, log_dir=None, save_every_model=False, device='cuda', dtype=torch.float64, save_fmaps=False, metrics=['energy_MAE','forces_MAE'])
_, weights = trainer.fit(model)
assert torch.isfinite(weights).all() and weights.norm() > 0
model.rf.weights = weights[:1].detach().clone()
model.save(root/'fitted.pt')
print('Fitted once on two 8-atom frames; saved',root/'fitted.pt',flush=True)
