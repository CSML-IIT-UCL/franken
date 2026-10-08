# water-transfer

- **Train:** frames `[:100]` of the original `water` training split (100 frames, indices 0–99).
- **Val:** frames `[100:450]` of the original `water` training split (350 frames, indices 100–449).

Both splits correspond to solid water with temperature increasing with time.

Use the dataset name `water-transfer`. Its `train-0-100.xyz` and `val-100-450.xyz` files are generated in a separate `water-transfer/` data folder. The filenames include the frame ranges to avoid reusing cached files from the previous split definition.
