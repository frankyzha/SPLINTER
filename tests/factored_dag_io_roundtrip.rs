//! Integration tests for the `RSMY` `FactoredDag` serializer (v4 / v5 / v6).

use std::sync::Arc;

use multiway_solver::factored_dag_io::{
    deserialize_factored_dag, deserialize_factored_dag_full, serialize_factored_dag,
    serialize_factored_dag_v6, serialize_factored_dag_version, DagCodec, FitConfigBlob,
};
use multiway_solver::types::{
    CostEntry, DeltaBudget, FactoredDag, FactoredSplit, FactoredSplitPool, FactoredSubproblem,
    FeatureType, InlineChildren, RLeafInfo, SplitSpec,
};

fn arc_f64(v: &[f64]) -> Arc<[f64]> {
    Arc::from(v.to_vec().into_boxed_slice())
}
fn arc_u16(v: &[u16]) -> Arc<[u16]> {
    Arc::from(v.to_vec().into_boxed_slice())
}

fn mk_leaf(cost: f64, pred: u8, loss: f64, n: usize) -> FactoredSubproblem {
    FactoredSubproblem {
        leaf: Some((
            cost,
            RLeafInfo {
                prediction: pred,
                loss,
                n_samples: n,
            },
        )),
        leaf_minority: None,
        prefix_leaf: None,
        splits: Vec::new(),
        split_ids: Vec::new(),
        tree_count: 1,
        // Leaf-only subproblem's min_obj is the leaf cost.
        min_obj: cost,
        cost_entries: None,
    }
}

fn mk_dag_simple_contig() -> FactoredDag {
    let leaf0 = mk_leaf(0.10, 0, 0.10, 100);
    let leaf1 = mk_leaf(0.05, 1, 0.05, 200);

    let thresholds = arc_f64(&[0.5]);
    // SpecPool: specs live in `FactoredDag.spec_pool`; the split
    // carries only the canonical `spec_id`.
    let spec_pool = vec![SplitSpec {
        feature_index: 3,
        arity: 2,
        thresholds: Arc::clone(&thresholds),
        group_map: None,
    }];
    let split = FactoredSplit::new(0, InlineChildren::from_slice(&[0, 1]), 1);

    let root = FactoredSubproblem {
        leaf: None,
        leaf_minority: None,
        prefix_leaf: None,
        splits: vec![split],
        split_ids: Vec::new(),
        tree_count: 1,
        // root min_obj = sum of children min_obj = 0.10 + 0.05 = 0.15.
        min_obj: 0.15,
        cost_entries: None,
    };

    fixture_dag(
        vec![leaf0, leaf1, root],
        spec_pool,
        2,
        vec![CostEntry::new(0.15, 1)],
        1,
        0.15,
        0.18,
        0.05,
        0.005,
        DeltaBudget {
            delta_beam: 0.001,
            delta_compress: 0.0,
            delta_quantize: 0.0005,
            delta_criterion: 0.0,
            delta_total: 0.0015,
            ..DeltaBudget::default()
        },
        1e9,
    )
}

#[allow(clippy::too_many_arguments)]
fn fixture_dag(
    subproblems: Vec<FactoredSubproblem>,
    spec_pool: Vec<SplitSpec>,
    root_ref: u32,
    root_cost_schedule: Vec<CostEntry>,
    tree_count: u128,
    opt: f64,
    rashomon_bound: f64,
    theta: f64,
    reg: f64,
    delta_budget: DeltaBudget,
    cost_precision: f64,
) -> FactoredDag {
    use dashmap::DashMap;
    FactoredDag {
        subproblems,
        root_ref,
        root_cost_schedule,
        tree_count,
        opt,
        rashomon_bound,
        theta,
        reg,
        delta_budget,
        cost_precision,
        keyed_costs: false,
        exact_cost_keys: false,
        split_pool: FactoredSplitPool::new(),
        spec_pool,
        inline_leaves: Vec::new(),
        inline_leaf_costs: Vec::new(),
        inline_leaves_minority: Vec::new(),
        inline_leaf_minority_costs: Vec::new(),
        pool_total_bytes: 0,
        pool_unique_split_count: 0,
        pool_dedup_hits: 0,
        pool_total_pushed: 0,
        schedule_intern_unique: 0,
        schedule_intern_attempts: 0,
        schedule_intern_hits: 0,
        schedule_intern_bytes: 0,
        arena_total_pushed: 0,
        arena_unique: 0,
        arena_dedup_hits: 0,
        arena_bytes: 0,
        schedule_cache: Arc::new(DashMap::new()),
        wide_schedule_cache: Default::default(),
        n_quotient_hash_collisions: 0,
        count_saturated: false,
    }
}

fn mk_dag_with_shape() -> FactoredDag {
    let thresholds = arc_f64(&[0.25, 0.75]);
    let group_map = arc_u16(&[0, 1, 0]);

    let leaf_a = mk_leaf(0.20, 0, 0.20, 50);
    let leaf_b = mk_leaf(0.15, 1, 0.15, 75);

    // Two distinct specs (feature 0 / feature 1) SHARING the thresholds and
    // group_map Arcs — exercises the serializer's Arc dedup across pool
    // entries (see `arc_dedup_preserves_sharing`).
    let spec_pool: Vec<SplitSpec> = (0u16..2)
        .map(|fi| SplitSpec {
            feature_index: fi,
            arity: 2,
            thresholds: Arc::clone(&thresholds),
            group_map: Some(Arc::clone(&group_map)),
        })
        .collect();
    let make_split =
        |spec_id: u32| FactoredSplit::new(spec_id, InlineChildren::from_slice(&[0, 1]), 1);

    let root = FactoredSubproblem {
        leaf: Some((
            0.42,
            RLeafInfo {
                prediction: 0,
                loss: 0.42,
                n_samples: 125,
            },
        )),
        leaf_minority: None,
        prefix_leaf: None,
        splits: vec![make_split(0), make_split(1)],
        split_ids: Vec::new(),
        tree_count: 3,
        // min(0.42 leaf, 0.20+0.15 split) = 0.35
        min_obj: 0.35,
        cost_entries: None,
    };

    fixture_dag(
        vec![leaf_a, leaf_b, root],
        spec_pool,
        2,
        vec![CostEntry::new(0.35, 2), CostEntry::new(0.42, 1)],
        3,
        0.35,
        0.45,
        0.07,
        0.005,
        DeltaBudget {
            delta_beam: 0.003,
            delta_compress: 0.02,
            delta_quantize: 0.004,
            delta_criterion: 0.0,
            delta_total: 0.027,
            ..DeltaBudget::default()
        },
        1e7,
    )
}

fn assert_dag_eq(a: &FactoredDag, b: &FactoredDag) {
    assert_eq!(a.subproblems.len(), b.subproblems.len());
    assert_eq!(a.root_ref, b.root_ref);
    assert_eq!(a.tree_count, b.tree_count);
    assert_eq!(a.opt.to_bits(), b.opt.to_bits());
    assert_eq!(a.rashomon_bound.to_bits(), b.rashomon_bound.to_bits());
    assert_eq!(a.theta.to_bits(), b.theta.to_bits());
    assert_eq!(a.reg.to_bits(), b.reg.to_bits());
    assert_eq!(a.cost_precision.to_bits(), b.cost_precision.to_bits());
    assert_eq!(a.keyed_costs, b.keyed_costs);
    assert_eq!(a.exact_cost_keys, b.exact_cost_keys);
    assert_eq!(a.root_cost_schedule.len(), b.root_cost_schedule.len());
    for (x, y) in a.root_cost_schedule.iter().zip(&b.root_cost_schedule) {
        assert_eq!(x.cost.to_bits(), y.cost.to_bits());
        assert_eq!(x.tree_count(), y.tree_count());
    }
    for (sa, sb) in a.subproblems.iter().zip(&b.subproblems) {
        assert_eq!(sa.tree_count, sb.tree_count);
        match (&sa.leaf, &sb.leaf) {
            (None, None) => {}
            (Some((ca, ia)), Some((cb, ib))) => {
                assert_eq!(ca.to_bits(), cb.to_bits());
                assert_eq!(ia.prediction, ib.prediction);
                assert_eq!(ia.loss.to_bits(), ib.loss.to_bits());
                assert_eq!(ia.n_samples, ib.n_samples);
            }
            _ => panic!("leaf option mismatch"),
        }
        assert_eq!(sa.splits.len(), sb.splits.len());
        for (xa, xb) in sa.splits.iter().zip(&sb.splits) {
            // SpecPool: compare the RESOLVED specs (v6 renumbers
            // spec ids by use and drops unreferenced specs).
            assert_eq!(
                a.spec_pool[xa.spec_id as usize],
                b.spec_pool[xb.spec_id as usize]
            );
            assert_eq!(xa.tree_count(), xb.tree_count());
            assert_eq!(xa.children.len(), xb.children.len());
            // v4+: per-child schedules are not stored; just compare FSubRefs.
            for (ra, rb) in xa.children.iter().zip(&xb.children) {
                assert_eq!(ra, rb);
            }
        }
    }
}

#[test]
fn roundtrip_contig() {
    let dag = mk_dag_simple_contig();
    let bytes = serialize_factored_dag(&dag, &[], &[], None);
    let (loaded, names, types, fc) = deserialize_factored_dag(&bytes).unwrap();
    assert!(names.is_empty());
    assert!(types.is_empty());
    assert!(fc.is_none());
    assert_dag_eq(&dag, &loaded);
}

#[test]
fn roundtrip_shape_with_features() {
    let dag = mk_dag_with_shape();
    let names = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
    let types = vec![FeatureType::Num, FeatureType::Cat, FeatureType::Num];
    let bytes = serialize_factored_dag(&dag, &names, &types, None);
    let (loaded, ln, lt, fc) = deserialize_factored_dag(&bytes).unwrap();
    assert_eq!(ln, names);
    assert_eq!(lt, types);
    assert!(fc.is_none());
    assert_dag_eq(&dag, &loaded);
}

#[test]
fn roundtrip_with_fit_config() {
    let dag = mk_dag_with_shape();
    let fc = FitConfigBlob {
        max_depth: 3,
        max_k: 3,
        min_samples_leaf: 20,
        reg_alpha: 1.0,
        path_penalty: 0.02,
        beam_width: 4,
        transition_delta: 0.02,
        atom_merge_tolerance: 0.01,
        top_k: Some(50),
        teacher_alpha: 1.0,
        feature_exclusion: false,
        shape_mode: true,
        atom_mode: true,
        cleanup_mode: false,
        dp_criterion: "rush".to_string(),
        atom_merge_rule: "l1".to_string(),
        extras: vec![
            ("teacher_cache".to_string(), "\"lightgbm\"".to_string()),
            ("lgbm_random_state".to_string(), "42".to_string()),
        ],
    };
    let bytes = serialize_factored_dag(&dag, &[], &[], Some(&fc));
    let (loaded_dag, _, _, loaded_fc) = deserialize_factored_dag(&bytes).unwrap();
    assert_dag_eq(&dag, &loaded_dag);
    let lfc = loaded_fc.expect("fit_config not round-tripped");
    assert_eq!(lfc.max_depth, fc.max_depth);
    assert_eq!(lfc.max_k, fc.max_k);
    assert_eq!(lfc.min_samples_leaf, fc.min_samples_leaf);
    assert_eq!(lfc.path_penalty.to_bits(), fc.path_penalty.to_bits());
    assert_eq!(lfc.beam_width, fc.beam_width);
    assert_eq!(
        lfc.transition_delta.to_bits(),
        fc.transition_delta.to_bits()
    );
    assert_eq!(lfc.top_k, fc.top_k);
    assert_eq!(lfc.shape_mode, fc.shape_mode);
    assert_eq!(lfc.atom_mode, fc.atom_mode);
    assert_eq!(lfc.dp_criterion, fc.dp_criterion);
    assert_eq!(lfc.atom_merge_rule, fc.atom_merge_rule);
    assert_eq!(lfc.extras, fc.extras);
}

#[test]
fn arc_dedup_preserves_sharing() {
    let dag = mk_dag_with_shape();
    let bytes = serialize_factored_dag(&dag, &[], &[], None);
    let (loaded, _, _, _) = deserialize_factored_dag(&bytes).unwrap();
    let root = &loaded.subproblems[loaded.root_ref as usize];
    assert_eq!(root.splits.len(), 2);
    // SpecPool: thresholds/group_map now live on the pool entries;
    // the two distinct specs (feature 0/1) must still share one Arc each.
    let spec0 = &loaded.spec_pool[root.splits[0].spec_id as usize];
    let spec1 = &loaded.spec_pool[root.splits[1].spec_id as usize];
    let t0 = &spec0.thresholds;
    let t1 = &spec1.thresholds;
    assert!(
        Arc::ptr_eq(t0, t1),
        "thresholds Arc dedup broke: t0 ptr {:p}, t1 ptr {:p}",
        Arc::as_ptr(t0),
        Arc::as_ptr(t1)
    );
    let g0 = spec0.group_map.as_ref().unwrap();
    let g1 = spec1.group_map.as_ref().unwrap();
    assert!(Arc::ptr_eq(g0, g1), "group_map Arc dedup broke");
}

#[test]
fn rejects_truncated() {
    let dag = mk_dag_simple_contig();
    let mut bytes = serialize_factored_dag(&dag, &[], &[], None);
    bytes.truncate(bytes.len() - 1);
    let err = deserialize_factored_dag(&bytes).unwrap_err();
    assert!(
        err.contains("trailer")
            || err.contains("EOF")
            || err.contains("CRC")
            || err.contains("sentinel"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn rejects_corrupted_byte() {
    let dag = mk_dag_simple_contig();
    let mut bytes = serialize_factored_dag(&dag, &[], &[], None);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    let err = deserialize_factored_dag(&bytes).unwrap_err();
    assert!(err.contains("CRC32"), "unexpected error: {}", err);
}

#[test]
fn rejects_wrong_magic() {
    let dag = mk_dag_simple_contig();
    let mut bytes = serialize_factored_dag(&dag, &[], &[], None);
    let v2_magic: u32 = 0x5253_4D58;
    bytes[..4].copy_from_slice(&v2_magic.to_le_bytes());
    // Recompute trailing CRC so the magic-check error fires before CRC fails.
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = !0;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }
    let trailer_off = bytes.len() - 8;
    let new_crc = crc32(&bytes[..trailer_off]);
    bytes[trailer_off..trailer_off + 4].copy_from_slice(&new_crc.to_le_bytes());
    let err = deserialize_factored_dag(&bytes).unwrap_err();
    assert!(err.contains("magic"), "unexpected error: {}", err);
}

#[test]
fn roundtrip_keyed_costs_and_legacy_files() {
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = !0;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }
    for (keyed, exact) in [(false, false), (true, false), (true, true)] {
        let mut dag = mk_dag_simple_contig();
        dag.keyed_costs = keyed;
        dag.exact_cost_keys = exact;
        let bytes = serialize_factored_dag(&dag, &[], &[], None);
        let (loaded, _, _, _) = deserialize_factored_dag(&bytes).unwrap();
        assert_eq!(loaded.keyed_costs, keyed);
        assert_eq!(loaded.exact_cost_keys, exact);
        assert_dag_eq(&dag, &loaded);
    }
    // A file written before the keyed-costs header bit existed (flag bit 3
    // clear, CRC valid) loads with float comparisons.
    let mut dag = mk_dag_simple_contig();
    dag.keyed_costs = true;
    let mut bytes = serialize_factored_dag(&dag, &[], &[], None);
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]) & !(1u16 << 3);
    bytes[6..8].copy_from_slice(&flags.to_le_bytes());
    let trailer_off = bytes.len() - 8;
    let new_crc = crc32(&bytes[..trailer_off]);
    bytes[trailer_off..trailer_off + 4].copy_from_slice(&new_crc.to_le_bytes());
    let (loaded, _, _, _) = deserialize_factored_dag(&bytes).unwrap();
    assert!(!loaded.keyed_costs);
}

/// Format version of serialized bytes (header u16 at offset 4).
fn version_of(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[4], bytes[5]])
}

/// The default writer emits v6; v5 and v4 stay writable and readable, and
/// all three load to the same DAG (fixtures: float costs = stored counts,
/// keyed = counts recomputed on load). The 3-vertex fixtures are too small
/// for v6 to beat v5 (stream framing); sizes are checked on
/// `mk_dag_layered` instead.
#[test]
fn v6_default_v5_v4_all_roundtrip() {
    for (keyed, exact) in [(false, false), (true, false), (true, true)] {
        for dag in [mk_dag_simple_contig(), mk_dag_with_shape()] {
            let mut dag = dag;
            dag.keyed_costs = keyed;
            dag.exact_cost_keys = exact;
            let v6 = serialize_factored_dag(&dag, &[], &[], None);
            let v5 = serialize_factored_dag_version(&dag, &[], &[], None, 5).unwrap();
            let v4 = serialize_factored_dag_version(&dag, &[], &[], None, 4).unwrap();
            let v6e = serialize_factored_dag_version(&dag, &[], &[], None, 6).unwrap();
            assert_eq!(version_of(&v6), 6);
            assert_eq!(version_of(&v6e), 6);
            assert_eq!(version_of(&v5), 5);
            assert_eq!(version_of(&v4), 4);
            assert!(v5.len() < v4.len(), "v5 {} >= v4 {}", v5.len(), v4.len());
            for bytes in [&v6, &v6e, &v5, &v4] {
                let (l, _, _, _) = deserialize_factored_dag(bytes).unwrap();
                assert_dag_eq(&dag, &l);
            }
        }
    }
    assert!(serialize_factored_dag_version(&mk_dag_simple_contig(), &[], &[], None, 3).is_err());
    assert!(serialize_factored_dag_version(&mk_dag_simple_contig(), &[], &[], None, 7).is_err());
}

/// The default writer: v6 with DEFLATE streams (codec byte 1).
#[test]
fn default_writer_is_v6_deflate() {
    let dag = mk_dag_simple_contig();
    let b = serialize_factored_dag(&dag, &[], &[], None);
    assert_eq!(u16::from_le_bytes([b[4], b[5]]), 6);
    assert_eq!(b[24], 1);
    assert_eq!(
        b,
        serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::Deflate)
    );
}

/// Recompute the trailer CRC after editing bytes.
fn fix_crc(bytes: &mut [u8]) {
    let trailer_off = bytes.len() - 8;
    let crc = {
        let mut crc: u32 = !0;
        for &b in &bytes[..trailer_off] {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    };
    bytes[trailer_off..trailer_off + 4].copy_from_slice(&crc.to_le_bytes());
}

/// A keyed v5 / v6 file stores no counts: corrupting the recorded root
/// tree_count (CRC fixed up) makes the load-time recount check fail.
#[test]
fn v5_keyed_recount_is_checked() {
    let mut dag = mk_dag_simple_contig();
    dag.keyed_costs = true;
    dag.exact_cost_keys = true;
    for v in [5u16, 6] {
        let mut bytes = serialize_factored_dag_version(&dag, &[], &[], None, v).unwrap();
        // METADATA starts at byte 32 with the u128 tree_count.
        bytes[32] ^= 0x02;
        fix_crc(&mut bytes);
        let err = deserialize_factored_dag(&bytes).unwrap_err();
        assert!(err.contains("recount"), "v{}: unexpected error: {}", v, err);
    }
}

/// (u128 counts) Split / vertex / root counts above u64 round-trip through
/// v4 (fixed u128), v5 and v6 (u128 varint), and a count at the u128::MAX
/// clamp sets `count_saturated` on load.
#[test]
fn u128_split_counts_roundtrip_v4_v5() {
    use multiway_solver::rashomon_dag::count_chain_fixture;
    let dag = count_chain_fixture(7, 0.001);
    let root_split_tc = dag.subproblems[dag.root_ref as usize].splits[0].tree_count();
    assert!(root_split_tc > u64::MAX as u128);
    assert!(!dag.count_saturated);
    let v6 = serialize_factored_dag(&dag, &[], &[], None);
    let v5 = serialize_factored_dag_version(&dag, &[], &[], None, 5).unwrap();
    let v4 = serialize_factored_dag_version(&dag, &[], &[], None, 4).unwrap();
    assert_eq!(version_of(&v6), 6);
    assert_eq!(version_of(&v5), 5);
    for bytes in [&v6, &v5, &v4] {
        let (l, _, _, _) = deserialize_factored_dag(bytes).unwrap();
        assert_dag_eq(&dag, &l);
        assert_eq!(
            l.subproblems[l.root_ref as usize].splits[0].tree_count(),
            root_split_tc
        );
        assert!(!l.count_saturated);
    }

    let sat = count_chain_fixture(8, 0.001);
    assert!(sat.count_saturated);
    for v in [4u16, 5, 6] {
        let bytes = serialize_factored_dag_version(&sat, &[], &[], None, v).unwrap();
        let (l, _, _, _) = deserialize_factored_dag(&bytes).unwrap();
        assert_eq!(l.tree_count, u128::MAX);
        assert!(l.count_saturated, "v{} load lost count_saturated", v);
    }
}

/// The keyed chain fixture (no stored counts: v6 recounts on load) gives
/// the same counts and saturation flag as the float one.
#[test]
fn u128_counts_v6_keyed_recount() {
    use multiway_solver::rashomon_dag::count_chain_fixture;
    for (levels, saturated) in [(7usize, false), (8, true)] {
        let mut dag = count_chain_fixture(levels, 0.001);
        dag.keyed_costs = true;
        dag.exact_cost_keys = true;
        let bytes = serialize_factored_dag_version(&dag, &[], &[], None, 6).unwrap();
        // FLAG_COUNTS_STORED (bit 5) clear: counts come from the recount.
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]) & (1 << 5), 0);
        let (l, _, _, _) = deserialize_factored_dag(&bytes).unwrap();
        let (l5, _, _, _) = deserialize_factored_dag(
            &serialize_factored_dag_version(&dag, &[], &[], None, 5).unwrap(),
        )
        .unwrap();
        assert_eq!(l.tree_count, l5.tree_count);
        assert_eq!(l.count_saturated, saturated);
        assert_eq!(l5.count_saturated, saturated);
        assert_dag_eq(&l5, &l);
    }
}

/// v6 METADATA_V6 fields after leaf_reg: (leaf_denom, n_train, root_ref,
/// n_leaf_escapes).
fn v6_meta(bytes: &[u8]) -> [u64; 4] {
    let mut pos = 32 + 72 + 8;
    let mut out = [0u64; 4];
    for o in out.iter_mut() {
        let mut shift = 0;
        loop {
            let b = bytes[pos];
            pos += 1;
            *o |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
    }
    out
}

/// Leaves whose (cost, loss) are not (k / D, k / D + leaf_reg) go through
/// the escape table and still round-trip bit-exactly.
#[test]
fn v6_leaf_escape_roundtrip() {
    let mut dag = mk_dag_simple_contig();
    // cost != loss + reg, and a loss that is not k / D for the chosen D.
    dag.subproblems[0] = mk_leaf(0.10, 0, 0.0333333, 7);
    dag.subproblems[1] = mk_leaf(0.05, 1, std::f64::consts::PI / 97.0, 200);
    for codec in [DagCodec::None, DagCodec::Deflate] {
        let bytes = serialize_factored_dag_v6(&dag, &[], &[], None, Some(300), codec);
        assert!(
            v6_meta(&bytes)[3] >= 1,
            "no leaf escaped: {:?}",
            v6_meta(&bytes)
        );
        let l = deserialize_factored_dag_full(&bytes).unwrap();
        assert_eq!(l.n_train, Some(300));
        assert_dag_eq(&dag, &l.dag);
    }
    // All leaves exact under D = 1000 (the caller's n_train).
    let mut dag = mk_dag_simple_contig();
    dag.subproblems[0] = mk_leaf(0.017 + 0.005, 0, 0.017, 7);
    dag.subproblems[1] = mk_leaf(0.333 + 0.005, 1, 0.333, 200);
    let bytes = serialize_factored_dag_v6(&dag, &[], &[], None, Some(1000), DagCodec::None);
    assert_eq!(
        v6_meta(&bytes)[3],
        0,
        "unexpected escapes: {:?}",
        v6_meta(&bytes)
    );
    assert_eq!(v6_meta(&bytes)[1], 1000);
    let l = deserialize_factored_dag_full(&bytes).unwrap();
    assert_dag_eq(&dag, &l.dag);
    let none = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::None);
    assert_eq!(deserialize_factored_dag_full(&none).unwrap().n_train, None);
}

/// Codec none and DEFLATE load to the same DAG; on the layered fixture
/// DEFLATE is smaller than none and v6 smaller than v5.
#[test]
fn v6_codec_none_and_deflate_identical_dag() {
    for dag in [
        mk_dag_simple_contig(),
        mk_dag_with_shape(),
        mk_dag_layered(false),
        mk_dag_layered(true),
    ] {
        let a = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::None);
        let b = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::Deflate);
        assert_eq!(a[24], 0);
        assert_eq!(b[24], 1);
        let (la, _, _, _) = deserialize_factored_dag(&a).unwrap();
        let (lb, _, _, _) = deserialize_factored_dag(&b).unwrap();
        assert_dag_eq(&la, &lb);
        assert_dag_iso(&dag, &la);
    }
    let dag = mk_dag_layered(false);
    let none = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::None);
    assert_eq!(
        v6_meta(&none)[3],
        0,
        "layered leaves escaped: {:?}",
        v6_meta(&none)
    );
    let defl = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::Deflate);
    let v5 = serialize_factored_dag_version(&dag, &[], &[], None, 5).unwrap();
    assert!(
        defl.len() < none.len(),
        "deflate {} >= none {}",
        defl.len(),
        none.len()
    );
    assert!(
        defl.len() < v5.len(),
        "v6 {} >= v5 {}",
        defl.len(),
        v5.len()
    );
}

/// v6 renumbers vertices (leaf-only first, internal in post-order) and
/// specs; on a scrambled, partly unreachable, pooled arena the loaded DAG
/// is isomorphic to the original with identical split / child order,
/// and v5 / v6 loads agree on the counts.
#[test]
fn v6_renumbering_roundtrip_layered() {
    for pooled in [false, true] {
        let dag = mk_dag_layered(pooled);
        let v6 = serialize_factored_dag(&dag, &[], &[], None);
        let l6 = deserialize_factored_dag_full(&v6).unwrap();
        assert_dag_iso(&dag, &l6.dag);
        // Unreferenced spec dropped.
        assert_eq!(l6.dag.spec_pool.len(), dag.spec_pool.len() - 1);
        let (l5, _, _, _) = deserialize_factored_dag(
            &serialize_factored_dag_version(&dag, &[], &[], None, 5).unwrap(),
        )
        .unwrap();
        assert_dag_iso(&l5, &l6.dag);
        // Re-encoding the loaded DAG is a fixed point.
        let again = serialize_factored_dag(&l6.dag, &[], &[], None);
        assert_eq!(again, v6);
    }
}

/// Corrupt v6 streams (CRC fixed up) are refused with an error, never a
/// panic.
#[test]
fn v6_rejects_bad_refs() {
    let dag = mk_dag_layered(false);
    let clean = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::None);
    let deflated = serialize_factored_dag_v6(&dag, &[], &[], None, None, DagCodec::Deflate);
    let mut n_err = 0;
    for base in [&clean, &deflated] {
        // Every byte between METADATA and the trailer tail, flipped one at a time.
        let body_end = base.len() - 8;
        for off in (32 + 72)..body_end {
            let mut bytes = base.clone();
            bytes[off] ^= 0x5A;
            fix_crc(&mut bytes);
            let r = std::panic::catch_unwind(|| deserialize_factored_dag(&bytes).map(|_| ()));
            let r = r.unwrap_or_else(|_| panic!("decoder panicked on a flipped byte at {}", off));
            if r.is_err() {
                n_err += 1;
            }
        }
    }
    assert!(n_err > 0);
    // Targeted: an internal ref delta of 0.
    let msgs: Vec<String> = (32 + 72..clean.len() - 8)
        .filter_map(|off| {
            let mut bytes = clean.clone();
            bytes[off] = 0;
            fix_crc(&mut bytes);
            deserialize_factored_dag(&bytes).err()
        })
        .collect();
    assert!(
        msgs.iter()
            .any(|m| m.contains("out of bounds") || m.contains("delta")),
        "no bounds error among {:?}",
        &msgs[..msgs.len().min(5)]
    );
}

/// Files written by the pre-v6 writer (`tests/data/`, generated from
/// `mk_dag_with_shape` with feature names) still load to the same DAG.
#[test]
fn old_format_fixture_files_load() {
    let names = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
    let types = vec![FeatureType::Num, FeatureType::Cat, FeatureType::Num];
    for (file, v) in [("dag_v5_fixture.bin", 5u16), ("dag_v4_fixture.bin", 4)] {
        let path = format!("{}/tests/data/{}", env!("CARGO_MANIFEST_DIR"), file);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(version_of(&bytes), v);
        let (l, ln, lt, _) = deserialize_factored_dag(&bytes).unwrap();
        assert_eq!(ln, names);
        assert_eq!(lt, types);
        assert_dag_eq(&mk_dag_with_shape(), &l);
        // And re-saved as v6.
        let (l6, _, _, _) =
            deserialize_factored_dag(&serialize_factored_dag(&l, &ln, &lt, None)).unwrap();
        assert_dag_eq(&l, &l6);
    }
}

/// Splits of a vertex, inline or through the promoted pool.
fn splits_of<'a>(dag: &'a FactoredDag, sp: &'a FactoredSubproblem) -> Vec<&'a FactoredSplit> {
    if sp.split_ids.is_empty() {
        sp.splits.iter().collect()
    } else {
        sp.split_ids
            .iter()
            .map(|&i| &dag.split_pool.entries[i as usize])
            .collect()
    }
}

/// Structural equality up to vertex / spec renumbering: walks both DAGs
/// from the root in split / child order, requiring a consistent bijection
/// of reachable vertices and bit-equal leaves, specs and counts.
fn assert_dag_iso(a: &FactoredDag, b: &FactoredDag) {
    use std::collections::HashMap;
    assert_eq!(a.tree_count, b.tree_count);
    assert_eq!(a.opt.to_bits(), b.opt.to_bits());
    assert_eq!(a.reg.to_bits(), b.reg.to_bits());
    assert_eq!(a.keyed_costs, b.keyed_costs);
    assert_eq!(a.root_cost_schedule.len(), b.root_cost_schedule.len());
    for (x, y) in a.root_cost_schedule.iter().zip(&b.root_cost_schedule) {
        assert_eq!(x.cost.to_bits(), y.cost.to_bits());
        assert_eq!(x.tree_count(), y.tree_count());
    }
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut stack = vec![(a.root_ref, b.root_ref)];
    while let Some((va, vb)) = stack.pop() {
        if let Some(&m) = map.get(&va) {
            assert_eq!(m, vb, "vertex {} maps to {} and {}", va, m, vb);
            continue;
        }
        map.insert(va, vb);
        let (sa, sb) = (&a.subproblems[va as usize], &b.subproblems[vb as usize]);
        assert_eq!(sa.tree_count, sb.tree_count);
        match (&sa.leaf, &sb.leaf) {
            (None, None) => {}
            (Some((ca, ia)), Some((cb, ib))) => {
                assert_eq!(ca.to_bits(), cb.to_bits());
                assert_eq!(ia.prediction, ib.prediction);
                assert_eq!(ia.loss.to_bits(), ib.loss.to_bits());
                assert_eq!(ia.n_samples, ib.n_samples);
            }
            _ => panic!("leaf option mismatch at {} / {}", va, vb),
        }
        let (xs, ys) = (splits_of(a, sa), splits_of(b, sb));
        assert_eq!(xs.len(), ys.len());
        for (xa, xb) in xs.iter().zip(&ys) {
            assert_eq!(
                a.spec_pool[xa.spec_id as usize],
                b.spec_pool[xb.spec_id as usize]
            );
            assert_eq!(xa.tree_count(), xb.tree_count());
            assert_eq!(xa.children.len(), xb.children.len());
            for (&ca, &cb) in xa.children.iter().zip(xb.children.iter()) {
                stack.push((ca, cb));
            }
        }
    }
    let mut seen: Vec<u32> = map.values().copied().collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), map.len(), "vertex map is not injective");
}

/// ~200-vertex fixture: 40 leaf-only vertices (loss k / 997 + reg 0.01),
/// 4 internal layers of 40 (every other one with a leaf, 2-3 splits over
/// 5 features, arity 2-3, children from all lower layers), a root over the
/// top layer, an unreachable internal vertex and an unreferenced spec. The
/// arena order is scrambled (not children-first). `pooled` moves the
/// splits into a promoted `split_pool`. Counts are stored (float costs)
/// and arbitrary.
fn mk_dag_layered(pooled: bool) -> FactoredDag {
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move |m: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % m
    };
    let grid: Vec<f64> = (0..12).map(|i| i as f64 * 0.37 - 1.0).collect();
    let mut spec_pool: Vec<SplitSpec> = Vec::new();
    for f in 0..5u16 {
        for arity in [2u8, 3] {
            let start = (f as usize + arity as usize) % 6;
            let thr: Vec<f64> = (0..arity as usize - 1)
                .map(|j| grid[start + 3 * j])
                .collect();
            spec_pool.push(SplitSpec {
                feature_index: f,
                arity,
                thresholds: arc_f64(&thr),
                group_map: if f == 4 && arity == 3 {
                    Some(arc_u16(&[0, 1, 2]))
                } else {
                    None
                },
            });
        }
    }
    // Unreferenced.
    spec_pool.push(SplitSpec {
        feature_index: 7,
        arity: 2,
        thresholds: arc_f64(&[9.5]),
        group_map: None,
    });
    let width = 40usize;
    // Logical ids: layer 0 = leaves, layers 1..=4 internal, then root, then unreachable.
    let mut logical: Vec<FactoredSubproblem> = Vec::new();
    for i in 0..width {
        let k = next(997);
        let loss = k as f64 / 997.0;
        logical.push(mk_leaf(
            loss + 0.01,
            (i % 2) as u8,
            loss,
            1 + next(500) as usize,
        ));
    }
    let mk_internal = |next: &mut dyn FnMut(u64) -> u64, lo: usize, hi: usize, with_leaf: bool| {
        let n_splits = 2 + next(2) as usize;
        let splits: Vec<FactoredSplit> = (0..n_splits)
            .map(|_| {
                let spec_id = next(10) as u32;
                let arity = spec_pool[spec_id as usize].arity as usize;
                let ch: Vec<u32> = (0..arity)
                    .map(|_| (lo + next((hi - lo) as u64) as usize) as u32)
                    .collect();
                FactoredSplit::new(
                    spec_id,
                    InlineChildren::from_slice(&ch),
                    1 + next(1000) as u128,
                )
            })
            .collect();
        let loss = next(997) as f64 / 997.0;
        FactoredSubproblem {
            leaf: with_leaf.then(|| {
                (
                    loss + 0.01,
                    RLeafInfo {
                        prediction: 1,
                        loss,
                        n_samples: 600,
                    },
                )
            }),
            leaf_minority: None,
            prefix_leaf: None,
            splits,
            split_ids: Vec::new(),
            tree_count: 1 + next(10_000) as u128,
            min_obj: f64::INFINITY,
            cost_entries: None,
        }
    };
    for layer in 1..=4 {
        for j in 0..width {
            let v = mk_internal(&mut next, 0, layer * width, j % 2 == 0);
            logical.push(v);
        }
    }
    let root_logical = logical.len();
    let root = mk_internal(&mut next, 4 * width, 5 * width, false);
    logical.push(root);
    let unreachable = mk_internal(&mut next, 0, 5 * width, true);
    logical.push(unreachable);
    // Scramble the arena: arena[pos[l]] = logical[l].
    let n = logical.len();
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, next(i as u64 + 1) as usize);
    }
    let mut arena: Vec<Option<FactoredSubproblem>> = (0..n).map(|_| None).collect();
    for (l, mut sp) in logical.into_iter().enumerate() {
        for split in sp.splits.iter_mut() {
            for c in split.children.iter_mut() {
                *c = perm[*c as usize] as u32;
            }
        }
        arena[perm[l]] = Some(sp);
    }
    let mut subproblems: Vec<FactoredSubproblem> = arena.into_iter().map(|s| s.unwrap()).collect();
    let mut pool = FactoredSplitPool::new();
    if pooled {
        for sp in subproblems.iter_mut() {
            for split in sp.splits.drain(..) {
                sp.split_ids.push(pool.entries.len() as u32);
                pool.entries.push(split);
            }
        }
    }
    let mut dag = fixture_dag(
        subproblems,
        spec_pool,
        perm[root_logical] as u32,
        vec![CostEntry::new(0.3, 5), CostEntry::new(0.31, 1234)],
        1239,
        0.3,
        0.33,
        0.1,
        0.01,
        DeltaBudget::default(),
        1e9,
    );
    dag.split_pool = pool;
    dag
}

/// A v6 stream whose comp_len varint is ~u64::MAX (CRC fixed up) is an
/// error, not an overflow panic.
#[test]
fn v6_rejects_huge_comp_len() {
    let dag = mk_dag_layered(false);
    for codec in [DagCodec::None, DagCodec::Deflate] {
        let base = serialize_factored_dag_v6(&dag, &[], &[], None, None, codec);
        // Skip METADATA_V6: f64 leaf_reg + 4 varints, then the first
        // stream's raw_len varint.
        let mut pos = 32 + 72 + 8;
        for _ in 0..5 {
            while base[pos] & 0x80 != 0 {
                pos += 1;
            }
            pos += 1;
        }
        let mut end = pos;
        while base[end] & 0x80 != 0 {
            end += 1;
        }
        end += 1;
        let mut bytes = base[..pos].to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
        bytes.extend_from_slice(&base[end..]);
        fix_crc(&mut bytes);
        let r = std::panic::catch_unwind(|| deserialize_factored_dag(&bytes).map(|_| ()));
        let err = r.expect("decoder panicked on a huge comp_len").unwrap_err();
        assert!(
            err.contains("EOF") || err.contains("out of bounds"),
            "unexpected error: {}",
            err
        );
    }
}

/// A spec whose thresholds are not strictly ascending cannot use the v6
/// gap code: the default writer falls back to v5 bytes, which round-trip.
#[test]
fn v6_non_ascending_spec_falls_back_to_v5() {
    let mut dag = mk_dag_simple_contig();
    dag.spec_pool[0] = SplitSpec {
        feature_index: 3,
        arity: 3,
        thresholds: arc_f64(&[0.75, 0.25]),
        group_map: None,
    };
    dag.subproblems.insert(2, mk_leaf(0.07, 1, 0.07, 30));
    dag.subproblems[3].splits[0] = FactoredSplit::new(0, InlineChildren::from_slice(&[0, 1, 2]), 1);
    dag.root_ref = 3;
    let bytes = serialize_factored_dag(&dag, &[], &[], None);
    assert_eq!(version_of(&bytes), 5);
    let explicit = serialize_factored_dag_version(&dag, &[], &[], None, 6).unwrap();
    assert_eq!(explicit, bytes);
    let (l, _, _, _) = deserialize_factored_dag(&bytes).unwrap();
    assert_dag_eq(&dag, &l);
}
