#1. Train franken
    cd tests/lammps/metatomic/

    franken.autotune \
        --dataset-name water --max-train-samples 8 \
        --l2-penalty="(-10, -5, 5, log)" \
        --force-weight="(0.01, 0.99, 5, linear)" \
        --jac-chunk-size 16 \
        --run-dir "./" \
        --backbone=pet --pet.path-or-id "PET_MAD/xs_1.5" \
        --rf=gaussian --gaussian.num-rf "256" --gaussian.length-scale="[5.0,10.]"

#2. Compile

    ckpt_path=$(ls */best_ckpt.pt); ckpt_dir=${ckpt_path%/*}
    franken.wrap_metatomic --model_path="${ckpt_dir}"/best_ckpt.pt --dtype=float64
    ln -s ${ckpt_dir}/best_ckpt-metatomic.pt best_ckpt-metatomic.pt 