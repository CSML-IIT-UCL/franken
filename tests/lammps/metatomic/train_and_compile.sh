#1. Train franken
    cd tests/lammps/metatomic/

    franken.autotune \
        --train-path ../train4.xyz \
        --l2-penalty="1e-8" \
        --force-weight="0.999" \
        --jac-chunk-size 16 \
        --run-dir "./" \
        --backbone=pet --pet.path-or-id "PET_MAD/xs_1.5" \
        --rf=gaussian --gaussian.num-rf "256" --gaussian.length-scale=5.

#2. Compile

    ckpt_path=$(ls */best_ckpt.pt); ckpt_dir=${ckpt_path%/*}
    franken.wrap_metatomic --model_path="${ckpt_dir}"/best_ckpt.pt --dtype=float64
    ln -s ${ckpt_dir}/best_ckpt-metatomic.pt best_ckpt-metatomic.pt 