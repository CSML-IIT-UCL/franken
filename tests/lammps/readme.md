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

    ```
2. Edit in-file with
    a. compiled model path
    b. data file path (PtH2O.lammpsdata)
3. Run lammps
    ```bash
    lmp -k on g 1 t 1 -sf kk -i $in_file -l $log_file
    ```