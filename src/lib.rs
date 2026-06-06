//! # ternary-shard-split
//!
//! Split ternary model weights across multiple devices for distributed training.
//! Since ternary weights pack 16 to a u32, sharding is clean: each shard gets
//! a multiple of 16 trits worth of packed data.

pub type Trit = i8;

/// A shard of ternary model weights.
#[derive(Debug, Clone)]
pub struct Shard {
    pub shard_id: usize,
    pub total_shards: usize,
    pub trits: Vec<Trit>,
    pub layer_name: String,
}

/// Split trits into N shards as evenly as possible.
pub fn split_even(trits: &[Trit], num_shards: usize) -> Vec<Shard> {
    assert!(num_shards > 0, "Need at least 1 shard");
    let chunk_size = trits.len() / num_shards;
    let remainder = trits.len() % num_shards;

    let mut shards = Vec::with_capacity(num_shards);
    let mut offset = 0;

    for i in 0..num_shards {
        let extra = if i < remainder { 1 } else { 0 };
        let size = chunk_size + extra;
        shards.push(Shard {
            shard_id: i,
            total_shards: num_shards,
            trits: trits[offset..offset + size].to_vec(),
            layer_name: String::new(),
        });
        offset += size;
    }
    shards
}

/// Split trits into shards aligned to 16-trit boundaries (for packed representation).
pub fn split_aligned(trits: &[Trit], num_shards: usize) -> Vec<Shard> {
    assert!(num_shards > 0);
    // Round each shard size up to multiple of 16
    let base = trits.len() / num_shards;
    let aligned_base = ((base + 15) / 16) * 16;

    let mut shards = Vec::with_capacity(num_shards);
    let mut offset = 0;

    for i in 0..num_shards {
        let size = if i == num_shards - 1 {
            trits.len() - offset // remaining goes to last shard
        } else {
            aligned_base.min(trits.len() - offset)
        };
        shards.push(Shard {
            shard_id: i,
            total_shards: num_shards,
            trits: trits[offset..offset + size].to_vec(),
            layer_name: String::new(),
        });
        offset += size;
    }
    shards
}

/// Split model layers across devices (layer-parallel sharding).
pub fn split_by_layers(layers: Vec<(&str, Vec<Trit>)>, num_devices: usize) -> Vec<Vec<Shard>> {
    let mut devices: Vec<Vec<Shard>> = (0..num_devices).map(|_| Vec::new()).collect();
    for (i, (name, trits)) in layers.into_iter().enumerate() {
        let device = i % num_devices;
        devices[device].push(Shard {
            shard_id: device,
            total_shards: num_devices,
            trits,
            layer_name: name.to_string(),
        });
    }
    devices
}

/// Merge shards back into a single weight vector.
pub fn merge(shards: &[Shard]) -> Vec<Trit> {
    let mut sorted: Vec<&Shard> = shards.iter().collect();
    sorted.sort_by_key(|s| s.shard_id);
    let mut result = Vec::new();
    for shard in sorted {
        result.extend_from_slice(&shard.trits);
    }
    result
}

/// Validate that shards reconstruct the original.
pub fn validate(shards: &[Shard], original: &[Trit]) -> bool {
    let merged = merge(shards);
    merged == original
}

/// Get size distribution across shards.
pub fn size_distribution(shards: &[Shard]) -> Vec<usize> {
    shards.iter().map(|s| s.trits.len()).collect()
}

/// Compute load imbalance factor (max_size / avg_size). 1.0 = perfect balance.
pub fn imbalance_factor(shards: &[Shard]) -> f64 {
    if shards.is_empty() { return 1.0; }
    let sizes = size_distribution(shards);
    let max = *sizes.iter().max().unwrap_or(&0) as f64;
    let avg = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
    if avg == 0.0 { 1.0 } else { max / avg }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_even_basic() {
        let trits = vec![1, -1, 0, 1, -1, 0, 1, -1, 0, 1];
        let shards = split_even(&trits, 3);
        assert_eq!(shards.len(), 3);
        // 10 / 3 = 3 with remainder 1 → sizes 4, 3, 3
        assert_eq!(shards[0].trits.len(), 4);
        assert_eq!(shards[1].trits.len(), 3);
        assert_eq!(shards[2].trits.len(), 3);
    }

    #[test]
    fn test_split_even_reconstructs() {
        let trits: Vec<Trit> = (0..100).map(|i| if i % 3 == 0 { 1 } else if i % 3 == 1 { -1 } else { 0 }).collect();
        let shards = split_even(&trits, 7);
        assert!(validate(&shards, &trits));
    }

    #[test]
    fn test_split_aligned_boundaries() {
        let trits = vec![1; 64]; // 64 trits
        let shards = split_aligned(&trits, 4);
        assert_eq!(shards.len(), 4);
        // Each shard should be aligned to 16
        for shard in &shards {
            assert_eq!(shard.trits.len() % 16, 0);
        }
        assert!(validate(&shards, &trits));
    }

    #[test]
    fn test_split_aligned_reconstructs() {
        let trits: Vec<Trit> = (0..97).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let shards = split_aligned(&trits, 3);
        assert!(validate(&shards, &trits));
    }

    #[test]
    fn test_split_by_layers() {
        let layers = vec![
            ("layer0", vec![1, -1, 0]),
            ("layer1", vec![0, 1, -1]),
            ("layer2", vec![1, 1, 1]),
            ("layer3", vec![-1, -1, -1]),
        ];
        let devices = split_by_layers(layers, 2);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].len(), 2); // layer0, layer2
        assert_eq!(devices[1].len(), 2); // layer1, layer3
        assert_eq!(devices[0][0].layer_name, "layer0");
        assert_eq!(devices[0][1].layer_name, "layer2");
    }

    #[test]
    fn test_merge_ordered() {
        let s1 = Shard { shard_id: 0, total_shards: 2, trits: vec![1, 1], layer_name: String::new() };
        let s2 = Shard { shard_id: 1, total_shards: 2, trits: vec![-1, -1], layer_name: String::new() };
        assert_eq!(merge(&[s2, s1]), vec![1, 1, -1, -1]); // sorts by shard_id
    }

    #[test]
    fn test_imbalance_perfect() {
        let shards = vec![
            Shard { shard_id: 0, total_shards: 2, trits: vec![1; 50], layer_name: String::new() },
            Shard { shard_id: 1, total_shards: 2, trits: vec![1; 50], layer_name: String::new() },
        ];
        assert!((imbalance_factor(&shards) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_imbalance_uneven() {
        let shards = vec![
            Shard { shard_id: 0, total_shards: 2, trits: vec![1; 90], layer_name: String::new() },
            Shard { shard_id: 1, total_shards: 2, trits: vec![1; 10], layer_name: String::new() },
        ];
        assert!(imbalance_factor(&shards) > 1.5);
    }

    #[test]
    fn test_single_shard() {
        let trits = vec![1, -1, 0];
        let shards = split_even(&trits, 1);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].trits, trits);
    }
}
