# Franken+LAMMPS (metatomic)

1. Train franken
    ```bash
    PYTHONPATH='.' python franken/autotune/script.py \
        --dataset-name water --max-train-samples 64 \
        --l2-penalty="(-10, -5, 5, log)" \
        --force-weight="(0.01, 0.99, 5, linear)" \
        --jac-chunk-size 16 \
        --backbone=pet --pet.path-or-id "PET_MAD/xs_1.5" \
        --rf=gaussian --gaussian.num-rf "256" --gaussian.length-scale="[5.0,10.]"
    ```

2. Compile
    ```bash
    ckpt_path=$(ls */best_ckpt.pt); ckpt_dir=${ckpt_path%/*}
    PYTHONPATH='.' python franken/calculators/metatomic_inf_wrap.py --model_path="${ckpt_dir}"/best_ckpt.pt --dtype=float64
    ln -s ${ckpt_dir}/best_ckpt-metatomic.pt best_ckpt-metatomic.pt 
    ```

3. Run lammps
    ```bash
    in_file=lammps_metatomic.in
    log_file=lammps_metatomic.log

    # if compiled with GPU (kokkos)
    lmp -suffix kk -k on g 1 -i $in_file -l $log_file
    # else
    lmp -i $in_file -l $log_file
    ```


*useful resources*
 - Metatomic LAMMPS github: https://github.com/metatensor/lammps/blob/metatomic/src/ML-METATOMIC/pair_metatomic.cpp
 - Metatomic LAMMPS documentation: https://docs.metatensor.org/metatomic/latest/engines/lammps.html