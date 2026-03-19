# Timings on bulk H2O transfer-learning

Problem settings: 128 training samples, 8192 random fourier features with the multiscale Gaussian kernel. Small HP search over force weight and lambda.

Hardware: NVIDIA RTX 6000 Ada GPU

## MACE baselines

### MACE MP
 - `mace_mp/small` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.990
     ```
   - Accuracy
     ```
     forces MAE: 16.337
     energy MAE: 0.237
     ```
   - Timings
     ```
     Cov/Coeff time (s): 402.8
     ```
 - `mace_mp/medium` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.990
     ```
   - Accuracy
     ```
     forces MAE: 15.176
     energy MAE: 0.361
     ```
   - Timings
     ```
     Cov/Coeff time (s): 1038.9
     ```

### MACE MH
 - `mace_mh/0` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.500
     ```
   - Accuracy
     ```
     forces MAE: 9.422
     energy MAE: 0.188
     ```
   - Timings
     ```
     Cov/Coeff time (s): 1025.0
     ```
 - `mace_mh/1` backbone - changed jac size to 28 because auto=32 was giving OOM
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.500
     ```
   - Accuracy
     ```
     forces MAE: 6.773
     energy MAE: 0.095
     ```
   - Timings
     ```
     Cov/Coeff time (s): 1941.5
     ```

### MACE OFF
 - `mace_off/small` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.255
     ```
   - Accuracy
     ```
     forces MAE: 24.903
     energy MAE: 0.287
     ```
   - Timings
     ```
     Cov/Coeff time (s): 196.5
     ```
 - `mace_off/medium` backbone - changed jac size from 50 because auto=64 was giving OOM
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.255
     ```
   - Accuracy
     ```
     forces MAE: 17.695
     energy MAE: 0.302
     ```
   - Timings
     ```
     Cov/Coeff time (s): 708.3
     ```
 - `mace_off/medium24` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.255
     ```
   - Accuracy
     ```
     forces MAE: 16.618
     energy MAE: 0.586
     ```
   - Timings
     ```
     Cov/Coeff time (s): 1032.6
     ```

## PET implementation branch

### PET MAD
 -  `PET_MAD/xs_1.5` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.500
     ```
   - Accuracy
     ```
     forces MAE: 40.595
     energy MAE: 0.800
     ```
   - Timings
     ```
     Cov/Coeff time (s): 130.9
     ```

 -  `PET_MAD/s_1.5` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.500
     ```
   - Accuracy
     ```
     forces MAE: 16.605
     energy MAE: 0.477
     ```
   - Timings
     ```
     Cov/Coeff time (s): 563.9
     ```

### MET OMAT
 - `PET_OMAT/xs_1.0` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.990
     ```
   - Accuracy
     ```
     forces MAE: 40.529
     energy MAE: 0.992
     ```
   - Timings
     ```
     Cov/Coeff time (s): 131.3
     ```
- `PET_OMAT/s_1.0` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.255
     ```
   - Accuracy
     ```
     forces MAE: 17.215
     energy MAE: 0.444
     ```
   - Timings
     ```
     Cov/Coeff time (s): 564.8
     ```
- `PET_OMAT/m_1.0` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.255
     ```
   - Accuracy
     ```
     forces MAE: 9.983
     energy MAE: 0.425
     ```
   - Timings
     ```
     Cov/Coeff time (s): 1970.1
     ```
- `PET_OMAT/l_1.0` backbone
   - Hyperparameters
     ```
     L2 penalty: 1.00e-10
     Force-weight: 0.255
     ```
   - Accuracy
     ```
     forces MAE: 7.685
     energy MAE: 0.163
     ```
   - Timings
     ```
     Cov/Coeff time (s): 4701.9
     ```
