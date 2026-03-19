*useful files*
```bash
in_file=lammps_metatomic.in
log_file=lammps_metatomic.log
```

0. Train franken
    ```bash
    PYTHONPATH='.' python franken/autotune/script.py \
        --dataset-name="PtH2O" \
        --max-train-samples=64 \
        --l2-penalty="(-10, -5, 5, log)" \
        --force-weight="(0.01, 0.99, 5, linear)" \
        --seed=42 \
        --jac-chunk-size=16 \
        --run-dir="franken/tests/lammps/" \
        --backbone=pet \
        --pet.path-or-id="PET_MAD/xs_1.5" \
        --rf=gaussian \
        --gaussian.num-rf=256 \
        --gaussian.length-scale="[10.0, 15.0]"
    ```

1. Compile
    ```bash
    PYTHONPATH='.' python franken/calculators/metatomic_inf_wrap.py --model_path=franken/tests/lammps/run_260312_093017_b4e1eaa9/best_ckpt.pt --dtype=float64
    ```
2. Edit in-file with
    a. compiled model path
    b. data file path (PtH2O.lammpsdata)
3. Run lammps
    ```bash
    lmp -i $in_file -l $log_file
    ```


*useful resources*
 - Metatomic LAMMPS github: https://github.com/metatensor/lammps/blob/metatomic/src/ML-METATOMIC/pair_metatomic.cpp
 - Metatomic LAMMPS documentation: https://docs.metatensor.org/metatomic/latest/engines/lammps.html