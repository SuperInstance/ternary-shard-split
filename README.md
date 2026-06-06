# ternary-shard-split

*Split the model. Keep the alignment.*

---

Sharding ternary model weights across devices for distributed training. Since ternary weights pack 16 to a u32, shards should be aligned to 16-trit boundaries for clean packed representation.

Provides: even splitting, aligned splitting (16-trit boundaries), layer-parallel splitting (assign layers round-robin to devices), merge back, validation, size distribution, and load imbalance measurement.

9 tests covering even split, aligned split, layer assignment, merge ordering, perfect vs imbalanced loads, single shard edge case.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
