# =========================
# 1. Imports and setup
# =========================
import os
import numpy as np
import torch
#import argparse
import pathlib
import re
import datetime

# ---- GPU configuration ----
# Force use of GPU 0 (adjust if needed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print('CUDA available =', cuda_available)

num_devices = torch.cuda.device_count()
print('# CUDA devices =', num_devices)


# =========================
# 2. ASE imports
# =========================

from ase.io import read, write
#from ase.md.langevin import Langevin

# =========================
# 3. Calculator imports
# =========================
from franken.rf.model import FrankenPotential
from franken.rf.les_model import LESFrankenPotential
from franken.calculators import FrankenCalculator

def get_newest_rundir(dir: pathlib.Path) -> pathlib.Path:
    """
    Find the newest run directory based on timestamp in the folder name.
    
    Folder format: run_DDMMYY_HHMMSS_randomstring
    Example: run_260723_214650_abc20294
    Args:
        dir: Path object pointing to the directory containing run folders
    Returns:
        Path to the newest run directory, or None if no matching folders found
    """
    if not dir.exists() or not dir.is_dir():
        raise ValueError(dir)
    
    pattern = re.compile(r'^run_(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_[a-zA-Z0-9]+$')
    valid_dirs = []
    for item in dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        try:
            day, month, year, hour, minute, second = match.groups()
            timestamp = datetime.datetime.strptime(
                f"{day}{month}{year}_{hour}{minute}{second}",
                "%d%m%y_%H%M%S"
            )
            valid_dirs.append((timestamp, item))
        except ValueError:
            continue
    if not valid_dirs:
        return None
    # Sort by timestamp descending and return the newest
    valid_dirs.sort(key=lambda x: x[0], reverse=True)
    return valid_dirs[0][1]

# =========================
# 4. Define calculator
# =========================
dataset_id=0 #change manually

franken_dir_SR = get_newest_rundir(pathlib.Path(f"franken_outputs/dimer_{dataset_id}"))
calc_SR  = FrankenCalculator(
    franken_dir_SR / "best_ckpt.pt",
    #model_class=FrankenPotential,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
franken_dir_LR = get_newest_rundir(pathlib.Path(f"les_outputs/dimer_{dataset_id}"))
calc_LR  = FrankenCalculator(
    franken_dir_LR / "best_ckpt.pt",
    #model_class=LESFrankenPotential,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)


# =========================
# 6. Read data
# =========================
input_file_path=f"dimer_{dataset_id}_dataset/"
input_file="all.xyz"
dataset=read(input_file_path+input_file,index=':',format='extxyz')

for snap in dataset:
    
    #DFT label
    snap.info["DFT_energy"]=snap.get_potential_energy()   
    snap.arrays["DFT_forces"]=snap.get_forces()
    
    # Overwrite stored results
    if "energy" in snap.info:
        del snap.info["energy"]
    if "forces" in snap.arrays:
        del snap.arrays["forces"]
    if "momenta" in snap.arrays:
        del snap.arrays["momenta"]
    
    snap.calc = calc_LR

    # trigger calculation
    calc_LR.calculate(
        snap,
        properties=["energy", "forces", "charges"]
    )
    #print(snap.calc.results.keys())
    
    #snap.set_calculator(calc)
    snap.info["LR_energy"]=snap.get_potential_energy()   
    snap.arrays["LR_forces"]=snap.get_forces()
    snap.arrays["charges"]=snap.calc.results["charges"]
    
    snap.calc = calc_SR

    # trigger calculation
    calc_SR.calculate(
        snap,
        properties=["energy", "forces"]
    )
    #print(snap.calc.results.keys())
    
    #snap.set_calculator(calc)
    snap.info["SR_energy"]=snap.get_potential_energy()   
    snap.arrays["SR_forces"]=snap.get_forces()

    
    output_file=f"predicted_{input_file}"
    write(output_file,snap,format='extxyz',append=True)
