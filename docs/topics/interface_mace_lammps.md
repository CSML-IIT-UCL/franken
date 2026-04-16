# MACE-LAMMPS

| **Software** | **Backbones** | **Usage** |
| --- | --- | --- |
| LAMMPS(+mace) | MACE | CLI: `franken.wrap_mace_lammps` |

If we have optimized a Franken model using a MACE backbone we can compile it for use with the [MACE](https://github.com/ACEsuit/lammps) fork of [LAMMPS](https://www.lammps.org/).

The basic steps required to run a Franken model with MACE-LAMMPS are:
 1. Compile the model using `franken/calculators/mace_inf_wrap.py`:
    ```bash
    franken.wrap_mace_lammps --model_path=<best_ckpt.pt>
    ```
    This export route is specific to the MACE LAMMPS fork. The compiled model will
    be saved in the same directory as the original model, with `-lammps` appended
    to the filename.
 2. Configure LAMMPS. The following lines are necessary, the second line should point to the compiled model from step 1 and describe the chemical species associated to the atom types in LAMMPS. 
    ```
    pair_style mace no_domain_decomposition
    pair_coeff * * <best_ckpt-lammps.pt> C H N O 
    ```
 3. Run LAMMPS-Mace. 

## Installing MACE-LAMMPS

This follows the [MACE guide](https://mace-docs.readthedocs.io/en/latest/guide/lammps.html) adapting it to the CINECA HPC Leonardo cluster.
This can be useful in case one wants to modify the Mace patch to LAMMPS. In particular, the following two files are important:
 - [https://github.com/ACEsuit/lammps/blob/mace/src/ML-MACE/pair_mace.cpp](https://github.com/ACEsuit/lammps/blob/mace/src/ML-MACE/pair_mace.cpp)
 - [https://github.com/ACEsuit/lammps/blob/mace/src/KOKKOS/pair_mace_kokkos.cpp](https://github.com/ACEsuit/lammps/blob/mace/src/KOKKOS/pair_mace_kokkos.cpp)

We will assume to start from directory `$BASE_DIR`
 1. ```git clone --branch=mace --depth=1 https://github.com/ACEsuit/lammps```
 2. download libtorch. For now keeping the default version as specified by MACE, but note that new versions exist!
    ```bash
    wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.2.0%2Bcu121.zip
    unzip libtorch-shared-with-deps-2.2.0+cu121.zip
    rm libtorch-shared-with-deps-2.2.0+cu121.zip
    mv libtorch libtorch-gpu
    ```
 3. Get a GPU node for compilation
    `srun -N 1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 -A <account> -p boost_usr_prod -t 00:30:00 --pty /bin/bash`
 4. Compile:
    1. Load modules
        ```bash
        module purge
        module load gcc/12.2.0
        module load gsl/2.7.1--gcc--12.2.0
        module load openmpi/4.1.6--gcc--12.2.0
        module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0
        module load openblas/0.3.24--gcc--12.2.0
        module load cuda/12.1
        module load intel-oneapi-mkl/2023.2.0
        ```
    2. Compile
        ```bash
        cd $BASE_DIR/lammps
        mkdir -p build-ampere
        cd build-ampere
        cmake \
            -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_INSTALL_PREFIX=$(pwd) \
            -D CMAKE_CXX_STANDARD=17 \
            -D CMAKE_CXX_STANDARD_REQUIRED=ON \
            -D BUILD_MPI=ON \
            -D BUILD_SHARED_LIBS=ON \
            -D PKG_KOKKOS=ON \
            -D Kokkos_ENABLE_CUDA=ON \
            -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
            -D Kokkos_ARCH_AMDAVX=ON \
            -D Kokkos_ARCH_AMPERE100=ON \
            -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch-gpu \
            -D PKG_ML-MACE=ON \
            ../cmake
        make -j 8
        make install
        ```
        The compiled binary is then at `$BASE_DIR/lammps/build-ampere/bin/lmp`. On the HPC cluster Leonardo you can find it pre-compiled here: 
        `/leonardo/pub/userexternal/lbonati1/software/lammps-mace/lammps/build-ampere-plumed/lmp`


## Running MACE-LAMMPS

This is just an example sbatch file which can be used to run MACE-LAMMPS. Edit it according to your needs. It uses the paths to the MACE-LAMMPS binary as available on the Leonardo cluster, and we will assume that LAMMPS has been configured in a file named `in.lammps`.

```bash
#!/bin/bash
#SBATCH --account=<account>
#SBATCH --partition=boost_usr_prod  # partition to be used
#SBATCH --time 00:30:00             # format: HH:MM:SS
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=1         # tasks out of 32
#SBATCH --gres=gpu:1                # gpus per node out of 4
#SBATCH --cpus-per-task=1           # Important: if > 1 kokkos complains.
############################

module purge
module load profile/base
module load gcc/12.2.0
module load gsl/2.7.1--gcc--12.2.0
module load openmpi/4.1.6--gcc--12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load cuda/12.1
module load intel-oneapi-mkl/2023.2.0

. /leonardo/pub/userexternal/lbonati1/software/lammps-mace/libtorch-gpu/sourceme.sh
. /leonardo/pub/userexternal/lbonati1/software/plumed/plumed2-2.9-gcc12/sourceme.sh

echo "setting env variable"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "running job"
in_file='in.lammps'
log_file='log.lammps'
lmp='/leonardo/pub/userexternal/lbonati1/software/lammps-mace/lammps/build-ampere-plumed/lmp'

srun $lmp -k on g 1 t ${SLURM_CPUS_PER_TASK} -sf kk -i $in_file -l $log_file

wait
```
