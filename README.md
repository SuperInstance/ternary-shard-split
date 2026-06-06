# ternary-shard-split

*Split ternary model weights across devices. Since {-1,0,+1} packs 16-to-a-u32, sharding is clean: every shard gets aligned data.*

## Why This Exists

Distributed training with ternary weights (BitNet b1.58 style) has an advantage nobody talks about: the sharding math is trivial. Binary weights need bit-level alignment tricks. Float16 needs padding to 2-byte boundaries. Ternary packs exactly 16 trits per u32, so splitting across N devices means each shard gets a clean multiple of 16 — no padding, no waste, no misaligned accesses.

This crate does the splitting, merging, and verification for distributed ternary training.

## Architecture

```
Original Weights: [tttttttttttttttttttttttttttttttt]
                                         ↓ split_even(N=4)
Shard 0: [tttttttt]    Shard 1: [tttttttt]
Shard 2: [tttttttt]    Shard 3: [tttttttt]
                                         ↓ merge_shards
Original Weights: [tttttttttttttttttttttttttttttttt] ✓
```

### Key Types

- **`Shard`** — A slice of ternary weights with metadata (shard_id, total_shards, layer_name)
- **`split_even(trits, N)`** — Split into N shards as evenly as possible (remainder distributed to first shards)
- **`split_aligned(trits, N, align)`** — Split respecting packing alignment (every shard is a multiple of `align` trits)
- **`merge_shards(shards)`** — Reconstruct original weights from shards
- **`shard_stats(shards)`** — Balance metrics (min/max/avg shard size, imbalance ratio)

### Design Decision: Even Distribution

`split_even` gives the first `remainder` shards one extra trit. This means shards differ by at most 1 trit. For a 1M-trit model split across 8 GPUs, that's a 0.0008% imbalance — negligible for any practical workload.

## Usage

```rust
use ternary_shard_split::*;

let weights: Vec<i8> = vec![-1, 0, 1, -1, 1, 0, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1];

// Split across 4 devices
let shards = split_even(&weights, 4);
assert_eq!(shards.len(), 4);

// Each shard has its metadata
assert_eq!(shards[0].shard_id, 0);
assert_eq!(shards[0].total_shards, 4);

// Merge back
let reconstructed = merge_shards(&shards).unwrap();
assert_eq!(reconstructed, weights);

// Check balance
let stats = shard_stats(&shards);
assert!(stats.imbalance_ratio() < 0.1);
```

## The Deeper Idea

Ternary sharding is a microcosm of the SuperInstance architecture: the {-1, 0, +1} representation is simple enough that distributed systems problems become trivial. When your fundamental unit is 2 bits (packed into integers), alignment, padding, and load balancing all have clean closed-form solutions.

This connects to `ternary-shard-merge` (the counterpart that merges shards with conflict resolution), `ternary-memory-pool` (device memory management), and `ternary-pipeline-parallel` (pipeline scheduling across sharded devices).

## Related Crates

- `ternary-pack` — Pack trits into u32 (the encoding this crate assumes)
- `ternary-shard-merge` — Merge shards with conflict resolution
- `ternary-pipeline-parallel` — Pipeline scheduling across sharded devices
- `ternary-tensor-parallel` — Tensor parallelism for ternary layers
- `ternary-memory-pool` — Device memory management for shards
