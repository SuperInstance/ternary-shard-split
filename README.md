# ternary-shard-split

Shard **ternary model weights** across multiple devices for distributed training. Since ternary weights pack 16 trits per `u32`, sharding is naturally aligned: each shard gets a multiple of 16 trits, maintaining packed-representation alignment with zero padding waste.

## Why It Matters

Ternary neural networks (e.g., BitNet, TernaryBERT) store weights as trits ∈ {-1, 0, +1}, packing 16 into a single 32-bit integer. When training across N GPUs, naive sharding can split mid-pack, corrupting the packed representation. This crate provides:

1. **`split_even`** — Equal-size shards (handle remainder)
2. **`split_aligned`** — 16-trit-aligned shards (packed-representation safe)
3. **`split_by_layers`** — Layer-parallel sharding (whole layers per device)
4. **`merge` / `validate`** — Reconstruct and verify

## How It Works

### Even Split

Given *T* trits and *N* shards, the first `T mod N` shards each get one extra trit:

```
chunk_size = ⌊T / N⌋
remainder  = T mod N

shard[i].size = chunk_size + (1 if i < remainder else 0)
```

**Imbalance factor:**

```
I = max_size / avg_size
```

For even split, `I ≤ 1 + 1/avg_size` — approaches 1.0 for large workloads.

**Complexity:** O(T) — single pass with index tracking.

### Aligned Split

Each shard's size is rounded up to a multiple of 16 (the pack width):

```
aligned_base = ⌈(T/N + 15) / 16⌉ × 16
shard[i].size = aligned_base  (for i < N-1)
shard[N-1].size = T - offset  (last shard gets remainder)
```

This ensures shards 0 through N-2 are perfectly aligned for packed operations. The last shard may be unaligned but contains all remaining data.

**Complexity:** O(T). No padding overhead for the first N-1 shards.

### Layer-Parallel Sharding

When layers are independent, assign round-robin to devices:

```
device(i) = layer[i] mod N
```

This distributes layers cyclically. With *L* layers and *N* devices, each device gets `⌈L/N⌉` or `⌊L/N⌋` layers.

**Complexity:** O(L) assignments.

### Merge and Validate

Merge sorts shards by `shard_id` and concatenates:

```
merged = concat(sort_by_id(shards).map(|s| s.trits))
```

Validate checks `merge(shards) == original`.

**Complexity:** O(S log S + T) for merge (sort + concat), where S = shard count.

## Quick Start

```rust
use ternary_shard_split::{split_even, split_aligned, split_by_layers, merge, validate};

let trits: Vec<i8> = (0..100).map(|i| if i % 3 == 0 {1} else if i%3==1 {-1} else {0}).collect();

// Even split into 7 shards
let shards = split_even(&trits, 7);
assert!(validate(&shards, &trits));

// Aligned split (16-trit boundaries)
let aligned = split_aligned(&trits, 4);
for s in &aligned[..aligned.len()-1] {
    assert_eq!(s.trits.len() % 16, 0); // aligned!
}

// Layer-parallel
let layers = vec![
    ("layer0", vec![1, -1, 0]),
    ("layer1", vec![0, 1, -1]),
    ("layer2", vec![1, 1, 1]),
    ("layer3", vec![-1, -1, -1]),
];
let devices = split_by_layers(layers, 2);
assert_eq!(devices.len(), 2);
```

## API

| Function | Description |
|----------|-------------|
| `split_even(trits, n)` | Equal-size shards |
| `split_aligned(trits, n)` | 16-trit-aligned shards |
| `split_by_layers(layers, n)` | Layer-parallel assignment |
| `merge(shards)` | Concatenate by shard_id |
| `validate(shards, original)` | Verify reconstruction |
| `size_distribution(shards)` | Vec of shard sizes |
| `imbalance_factor(shards)` | max / avg (1.0 = perfect) |

### Types

```rust
pub type Trit = i8;

pub struct Shard {
    pub shard_id: usize,
    pub total_shards: usize,
    pub trits: Vec<Trit>,
    pub layer_name: String,
}
```

## Architecture Notes

The **γ + η = C** invariant: *generation* (γ) is the split operation producing shard boundaries, *entropy* (η) is the load distribution (measured by `imbalance_factor`), and *conservation* (C) is the invariant that `merge(split(T, N)) == T` — no trit is lost or duplicated during distribution and reconstruction. The `validate` function explicitly checks C. The aligned split additionally preserves a structural invariant: packed-representation alignment for efficient GPU operations.

## References

- **Model parallelism:** Shoeybi, M. et al. "Megatron-LM" (2019)
- **Ternary weight packing:** Alemdar, H. et al. "Ternary Weight Networks" (2017), §3.1
- **Data parallel sharding:** Li, M. et al. "Scaling Distributed Machine Learning" (2014)
- **Load balancing theory:** Cybenko, G. "Dynamic Load Balancing" (1989)

## License

MIT
