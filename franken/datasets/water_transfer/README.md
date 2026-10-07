# water-transfer

- **Train:** the original `water` validation split; solid water at constant temperature.
- **Val:** the first 450 frames of the original `water` training split (`[:450]`, indices 0–449); solid water with temperature increasing with time.

Use the dataset name `water-transfer`. Its `train.xyz` and `val.xyz` files are generated in a separate `water-transfer/` data folder from the original water dataset.
