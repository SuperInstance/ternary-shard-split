# Ternary Shard Split — Distributed Training via Ternary Weight Sharding

**Ternary Shard Split** partitions ternary model weights across multiple devices for distributed training. Since ternary weights pack 16-per-u32 (2 bits each), sharding is naturally aligned: each shard receives a multiple of 16 trits, ensuring clean boundaries in the packed representation.

## Why It Matters

Large ternary models can exceed single-GPU memory despite being 16× smaller than FP32. Sharding distributes the model across devices, enabling training of models that would otherwise be impossible. The ternary packing makes sharding especially clean: unlike float weights where arbitrary split points can misalign with cache lines, ternary shards always start and end on 16-trit (1-u32) boundaries. This eliminates the partial-word handling that complicates binary and float sharding schemes.

## How It Works

### Even Split

`split_even(trits, n)` divides trits into n shards as evenly as possible. For N trits and n shards, each shard gets ⌈N/n⌉ or ⌊N/n⌋ trits. The first `N mod n` shards get one extra trit. O(N) to split.

### Aligned Split

`split_aligned(trits, n)` ensures each shard boundary falls on a 16-trit boundary. Each shard (except the last) gets a multiple of 16 trits. This is critical for packed representation: shards can be directly loaded as u32 arrays without bit manipulation.

```
aligned_base = ⌈(N / n + 15) / 16⌉ × 16
shard[0..n-1].len = aligned_base
shard[n-1].len = remaining (may be < aligned_base)
```

### Layer-Parallel Sharding

`split_by_layers(layers, n)` distributes entire model layers across devices. Layer i goes to device `i mod n`. This enables pipeline parallelism: device 0 computes layer 0, passes output to device 1 for layer 1, etc. Each device holds only its assigned layers.

### Shard Metadata

Each `Shard` carries:
- `shard_id`: Which shard this is (0..n-1)
- `total_shards`: Total number of shards
- `trits`: The actual ternary weight data
- `layer_name`: Which model layer (for layer-parallel)

### Reconstruction

Concatenating all shards in order reconstructs the original weight tensor. O(N) total.

## Quick Start

```rust
use ternary_shard_split::{split_even, split_aligned};

let trits: Vec<i8> = (0..1000).map(|i| match i % 3 { 0 => -1, 1 => 0, _ => 1 }).collect();

// Even split across 4 devices
let shards = split_even(&trits, 4);
assert_eq!(shards.len(), 4);
assert_eq!(shards.iter().map(|s| s.trits.len()).sum::<usize>(), 1000);

// Aligned split (16-trit boundaries)
let aligned = split_aligned(&trits, 4);
```

```bash
cargo add ternary-shard-split
```

## API

| Type / Function | Description |
|---|---|
| `Shard` | `{ shard_id, total_shards, trits, layer_name }` |
| `split_even(&[Trit], n)` | Even partition (O(N)) |
| `split_aligned(&[Trit], n)` | 16-trit aligned partition |
| `split_by_layers(layers, n)` | Layer-parallel distribution |

## Architecture Notes

Shard split enables multi-GPU training in **SuperInstance**. Each GPU holds a shard of the ternary model; gradients are exchanged between shards during training. The γ + η = C conservation holds per-shard: each shard's active weights (γ) plus zero weights (η) equals the shard size C. Global conservation follows from summing across shards. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Shazeer, Noam et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," *ICLR*, 2017.
- Rajbhandari, Samyam et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," *SC*, 2020.
| Huang, Yanping et al. "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," *NeurIPS*, 2019.



## Complexity Summary

| Operation | Time | Space |
|---|---|---|
| split_even(N trits, n shards) | O(N) | O(N) |
| split_aligned(N trits, n shards) | O(N) | O(N) |
| split_by_layers(L layers, n devices) | O(L) | O(L) |
| Reconstruction (concat shards) | O(N) | O(N) |

Aligned splitting adds O(16) padding overhead per shard in the worst case, negligible for models with millions of weights.

## License

MIT
