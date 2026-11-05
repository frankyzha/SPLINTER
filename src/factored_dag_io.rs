//! Binary serialization for `FactoredDag` (formats v4–v6, magic `RSMY`).
//!
//! The active Rashomon representation is the cross-product-free `FactoredDag`.
//! The older v2 (`RSMX`) and v3 formats are not readable.
//!
//! Layout: little-endian, sectioned, fixed-size records where possible. Arc
//! sharing is preserved by deduplicating `Arc<[f64]>` thresholds and
//! `Arc<[u16]>` group_maps into front tables and referencing them by u32 id.
//! The arena order in `dag.subproblems` is topological by construction
//! (`build_factored_recursive`), so writing in index order keeps `FSubRef`
//! values valid on reload.
//!
//! v5 (v4 still read and writable via
//! `serialize_factored_dag_version`) keeps the v4 header and metadata and
//! replaces the fixed records with LEB128 varints and shared tables:
//! content-deduplicated threshold / group_map tables, an interned spec
//! table (feature, arity, threshold id, group_map id), a leaf-value table
//! of distinct (cost, loss) f64 pairs (bit-exact; no per-vertex loss), one
//! split-record table (spec id + child vertex ids) referenced by index from
//! the vertices, and per vertex: leaf flag, prediction, leaf-value id,
//! n_samples, split-record ids. Keyed DAGs without build-time schedules,
//! prefix leaves or minority leaves store NO tree counts: the
//! loader recomputes min_obj, split / vertex counts and the root schedule
//! with the build's count pass (`rashomon_dag::recount_at_bound`) at the
//! stored beta and precision, and checks the stored root tree_count.
//! Other DAGs store varint counts (FLAG_COUNTS_STORED).
//!
//! v6 (default writer; the older v5 / v4 layouts stay readable and
//! writable via `serialize_factored_dag_version`) keeps the 32-byte header width, the
//! 72-byte METADATA block, FEATURE_META, FIT_CONFIG and the CRC trailer, and
//! stores the body as struct-of-arrays varint streams, each compressed on
//! its own (codec byte in the header: 0 = none, 1 = DEFLATE via `flate2`'s
//! pure-Rust backend, 2 = reserved; the writers use DEFLATE unless
//! `serialize_factored_dag_v6` is given `DagCodec::None`). Vertices are renumbered: leaf-only vertices (a leaf, no splits)
//! first, ranked by how often they are referenced, then the other vertices
//! in DFS post-order from the root, so an internal child is referenced by
//! the delta `parent position - child position` (>= 1). Split lists and
//! child order are copied verbatim and no code orders by vertex or spec id,
//! so tree_count, schedules and `get_tree(i)` order are unchanged. Specs are
//! ranked by use (unreferenced specs are dropped); thresholds are indices
//! into a per-feature f64 grid. A leaf value is stored as an integer k with
//! loss = k / leaf_denom and cost = loss + leaf_reg when that reproduces
//! both f64s bit-exactly, else as a raw (cost, loss) escape pair. Streams:
//! GRID, GMAPS, SPECS, VHDR (2 * n_splits + has_leaf per internal vertex),
//! PRED, KCODE, K_ESC, NSAMP, SPEC_IDS, TAGS (1 bit per child: leaf-only),
//! LREF, IREF [, COUNTS when FLAG_COUNTS_STORED]. The file also records the
//! caller's training-row count (`n_train`, 0 = unknown).
//!
//! A vertex's stored leaf `n_samples` (all versions) is the row count of
//! one support of its merged class: the quotient (`rashomon_dag`) merges
//! leaves with equal cost, prediction and loss, i.e. the same number of
//! misclassified rows, whatever their total row count (pure leaves of any
//! size merge). It is not the row count of every occurrence of the vertex
//! in an extracted tree; `PyRashomonDag.attach_training_data` restores
//! per-tree counts by routing the training rows.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::{
    CostEntry, DeltaBudget, FSubRef, FactoredDag, FactoredSplit, FactoredSubproblem, FeatureType,
    RLeafInfo, SplitSpec,
};

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

const MAGIC_V3: u32 = 0x5253_4D59; // "RSMY" little-endian
const VERSION_V3: u16 = 3;
/// v4: v3 stored per-edge cost schedules
/// on each `FactoredSplit`; v4 only stores `Vec<FSubRef>` and recomputes
/// schedules at extract time. Legacy v3 files are explicitly refused.
const VERSION_V4: u16 = 4;
/// Compact varint / shared-table format (see the module docs).
const VERSION_V5: u16 = 5;
/// Compressed struct-of-arrays stream format (see the module docs).
const VERSION_V6: u16 = 6;
/// v6 stream codecs (header byte). 2 is reserved for brotli.
const CODEC_NONE: u8 = 0;
const CODEC_DEFLATE: u8 = 1;
/// v6 stream count without / with the COUNTS stream.
const V6_N_STREAMS: u8 = 12;
const TRAILER_SENTINEL: u32 = 0xDEAD_BEEF;
const NO_GROUP_MAP: u32 = u32::MAX;

const FLAG_HAS_GROUP_MAPS: u16 = 1 << 0;
const FLAG_HAS_FEATURE_META: u16 = 1 << 1;
const FLAG_HAS_FIT_CONFIG: u16 = 1 << 2;
/// Counts were built in integer key space (`FactoredDag::keyed_costs`).
/// Files written before this bit existed leave it clear and load as float.
const FLAG_KEYED_COSTS: u16 = 1 << 3;
/// Keys are exact lattice keys (`FactoredDag::exact_cost_keys`); only set
/// together with FLAG_KEYED_COSTS. Older files leave it clear.
const FLAG_EXACT_COST_KEYS: u16 = 1 << 4;
/// v5 only: per-vertex / per-split tree counts and the root schedule are in
/// the file (DAGs the loader cannot recount); clear = recount on load.
const FLAG_COUNTS_STORED: u16 = 1 << 5;

/// Schema version for the optional FIT_CONFIG section. Independent of the
/// main format version so we can extend with new fields without bumping v3.
const FIT_CONFIG_SCHEMA_V1: u16 = 1;

// FIT_CONFIG bool_flags bits.
const FCFG_FEATURE_EXCLUSION: u32 = 1 << 0;
const FCFG_SHAPE_MODE: u32 = 1 << 1;
const FCFG_ATOM_MODE: u32 = 1 << 2;
const FCFG_CLEANUP_MODE: u32 = 1 << 5;

const HEADER_LEN: usize = 32;
const METADATA_LEN: usize = 72;

// ---------------------------------------------------------------------------
// Little-endian write helpers
// ---------------------------------------------------------------------------

fn put_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}
fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u128(buf: &mut Vec<u8>, v: u128) {
    put_u64(buf, v as u64);
    put_u64(buf, (v >> 64) as u64);
}
fn put_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
/// Unsigned LEB128 (7 bits per byte, high bit = continuation).
fn put_varint(buf: &mut Vec<u8>, mut v: u128) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

// ---------------------------------------------------------------------------
// Little-endian read cursor
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    /// Bytes left to read: an upper bound on the number of items a length
    /// field can still describe (every item takes at least one byte), used
    /// to cap header-driven preallocations.
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn need(&self, n: usize) -> Result<(), String> {
        // `n` may come from an untrusted length field: no `pos + n` overflow.
        if n > self.data.len().saturating_sub(self.pos) {
            return Err(format!(
                "unexpected EOF at byte {}: need {} more, have {}",
                self.pos,
                n,
                self.data.len() - self.pos
            ));
        }
        Ok(())
    }
    fn read_u8(&mut self) -> Result<u8, String> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn read_u16(&mut self) -> Result<u16, String> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }
    fn read_u32(&mut self) -> Result<u32, String> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn read_u64(&mut self) -> Result<u64, String> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn read_u128(&mut self) -> Result<u128, String> {
        let lo = self.read_u64()? as u128;
        let hi = self.read_u64()? as u128;
        Ok((hi << 64) | lo)
    }
    fn read_f32(&mut self) -> Result<f32, String> {
        self.need(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn read_f64(&mut self) -> Result<f64, String> {
        self.need(8)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn read_varint(&mut self) -> Result<u128, String> {
        let mut v: u128 = 0;
        let mut shift = 0u32;
        loop {
            let b = self.read_u8()?;
            if shift >= 128 || (shift == 126 && (b & 0x7F) > 0x03) {
                return Err(format!("varint overflow at byte {}", self.pos));
            }
            v |= ((b & 0x7F) as u128) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    fn read_varint_u64(&mut self) -> Result<u64, String> {
        u64::try_from(self.read_varint()?)
            .map_err(|_| format!("varint exceeds u64 at byte {}", self.pos))
    }
    fn read_varint_u32(&mut self) -> Result<u32, String> {
        u32::try_from(self.read_varint()?)
            .map_err(|_| format!("varint exceeds u32 at byte {}", self.pos))
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        self.need(n)?;
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// Tiny CRC32 (IEEE polynomial, table-free). Throughput is fine for files
// up to hundreds of MB; we'd promote to a table-based impl if profiles
// say otherwise.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Arc dedup tables (built during serialize, restored during deserialize)
// ---------------------------------------------------------------------------

struct DedupTables {
    thresholds: Vec<Arc<[f64]>>,
    threshold_id: HashMap<usize, u32>, // key = Arc::as_ptr() as usize
    group_maps: Vec<Arc<[u16]>>,
    group_map_id: HashMap<usize, u32>,
}

impl DedupTables {
    fn new() -> Self {
        Self {
            thresholds: Vec::new(),
            threshold_id: HashMap::new(),
            group_maps: Vec::new(),
            group_map_id: HashMap::new(),
        }
    }

    fn intern_thresholds(&mut self, arc: &Arc<[f64]>) -> u32 {
        let key = Arc::as_ptr(arc) as *const () as usize;
        if let Some(&id) = self.threshold_id.get(&key) {
            return id;
        }
        let id = self.thresholds.len() as u32;
        self.thresholds.push(Arc::clone(arc));
        self.threshold_id.insert(key, id);
        id
    }

    fn intern_group_map(&mut self, arc: &Arc<[u16]>) -> u32 {
        let key = Arc::as_ptr(arc) as *const () as usize;
        if let Some(&id) = self.group_map_id.get(&key) {
            return id;
        }
        let id = self.group_maps.len() as u32;
        self.group_maps.push(Arc::clone(arc));
        self.group_map_id.insert(key, id);
        id
    }
}

/// Resolve a subproblem's splits, dispatching via the pool when
/// the dag has been promoted (`split_pool` populated and `split_ids` set).
fn subproblem_splits<'a>(
    dag: &'a FactoredDag,
    sp: &'a FactoredSubproblem,
) -> Box<dyn Iterator<Item = &'a FactoredSplit> + 'a> {
    if !sp.split_ids.is_empty() {
        Box::new(
            sp.split_ids
                .iter()
                .map(move |&id| &dag.split_pool.entries[id as usize]),
        )
    } else {
        Box::new(sp.splits.iter())
    }
}

fn subproblem_n_splits(sp: &FactoredSubproblem) -> usize {
    if !sp.split_ids.is_empty() {
        sp.split_ids.len()
    } else {
        sp.splits.len()
    }
}

fn build_dedup(dag: &FactoredDag) -> DedupTables {
    let mut d = DedupTables::new();
    // Walk pool entries directly if promoted (saves N-fold
    // redundant Arc lookups for shared splits).
    if !dag.split_pool.entries.is_empty() {
        for split in &dag.split_pool.entries {
            let spec = &dag.spec_pool[split.spec_id as usize];
            d.intern_thresholds(&spec.thresholds);
            if let Some(gm) = &spec.group_map {
                d.intern_group_map(gm);
            }
        }
    } else {
        for sp in &dag.subproblems {
            for split in &sp.splits {
                let spec = &dag.spec_pool[split.spec_id as usize];
                d.intern_thresholds(&spec.thresholds);
                if let Some(gm) = &spec.group_map {
                    d.intern_group_map(gm);
                }
            }
        }
    }
    d
}

// ---------------------------------------------------------------------------
// FitConfig: optional self-describing record of the SolverConfig and Python
// kwargs the DAG was fit with. Lets a saved DAG be re-fit or audited without
// out-of-band metadata. Stored under FLAG_HAS_FIT_CONFIG.
// ---------------------------------------------------------------------------

/// Snapshot of the fit-time configuration, persisted alongside the DAG.
///
/// Numeric and enum-string fields cover the Rust SolverConfig surface that
/// affects refit reproducibility (the bits already serialized via theta /
/// reg / opt / cost_precision / delta_budget are not duplicated). The
/// `extras` map is for Python-only kwargs (`lgbm_random_state`,
/// `teacher_cache`, ...) — values are arbitrary
/// JSON strings, opaque to Rust.
#[derive(Clone, Debug, Default)]
pub struct FitConfigBlob {
    // Numeric core
    pub max_depth: i32,
    pub max_k: u32,
    pub min_samples_leaf: u32,
    pub reg_alpha: f64,
    pub path_penalty: f64,
    pub beam_width: u32,
    pub transition_delta: f64,
    pub atom_merge_tolerance: f64,
    pub top_k: Option<u32>,
    pub teacher_alpha: f64,

    // Boolean flags
    pub feature_exclusion: bool,
    pub shape_mode: bool,
    pub atom_mode: bool,
    pub cleanup_mode: bool,

    // Enum-as-string
    pub dp_criterion: String,
    pub atom_merge_rule: String,

    // Python extras: key → JSON-encoded value
    pub extras: Vec<(String, String)>,
}

fn write_fit_config(buf: &mut Vec<u8>, fc: &FitConfigBlob) {
    put_u16(buf, FIT_CONFIG_SCHEMA_V1);
    put_u16(buf, 0); // reserved

    put_u32(buf, fc.max_depth as u32);
    put_u32(buf, fc.max_k);
    put_u32(buf, fc.min_samples_leaf);
    put_f64(buf, fc.reg_alpha);
    put_f64(buf, fc.path_penalty);
    put_u32(buf, fc.beam_width);
    put_f64(buf, fc.transition_delta);
    put_f64(buf, fc.atom_merge_tolerance);
    put_u32(buf, fc.top_k.unwrap_or(u32::MAX));
    put_f64(buf, fc.teacher_alpha);

    let mut bool_flags: u32 = 0;
    if fc.feature_exclusion {
        bool_flags |= FCFG_FEATURE_EXCLUSION;
    }
    if fc.shape_mode {
        bool_flags |= FCFG_SHAPE_MODE;
    }
    if fc.atom_mode {
        bool_flags |= FCFG_ATOM_MODE;
    }
    if fc.cleanup_mode {
        bool_flags |= FCFG_CLEANUP_MODE;
    }
    put_u32(buf, bool_flags);

    write_lp_str(buf, &fc.dp_criterion);
    write_lp_str(buf, &fc.atom_merge_rule);

    put_u32(buf, fc.extras.len() as u32);
    for (k, v) in &fc.extras {
        write_lp_str(buf, k);
        write_lp_str(buf, v);
    }
}

fn write_lp_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    put_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

fn read_fit_config(cur: &mut Cursor) -> Result<FitConfigBlob, String> {
    let schema = cur.read_u16()?;
    let _reserved = cur.read_u16()?;
    if schema != FIT_CONFIG_SCHEMA_V1 {
        return Err(format!(
            "FitConfig schema {} not supported (this build reads v{})",
            schema, FIT_CONFIG_SCHEMA_V1
        ));
    }

    let max_depth = cur.read_u32()? as i32;
    let max_k = cur.read_u32()?;
    let min_samples_leaf = cur.read_u32()?;
    let reg_alpha = cur.read_f64()?;
    let path_penalty = cur.read_f64()?;
    let beam_width = cur.read_u32()?;
    let transition_delta = cur.read_f64()?;
    let atom_merge_tolerance = cur.read_f64()?;
    let top_k_raw = cur.read_u32()?;
    let top_k = if top_k_raw == u32::MAX {
        None
    } else {
        Some(top_k_raw)
    };
    let teacher_alpha = cur.read_f64()?;

    let bool_flags = cur.read_u32()?;
    let feature_exclusion = (bool_flags & FCFG_FEATURE_EXCLUSION) != 0;
    let shape_mode = (bool_flags & FCFG_SHAPE_MODE) != 0;
    let atom_mode = (bool_flags & FCFG_ATOM_MODE) != 0;
    let cleanup_mode = (bool_flags & FCFG_CLEANUP_MODE) != 0;

    let dp_criterion = read_lp_str(cur)?;
    let atom_merge_rule = read_lp_str(cur)?;

    let n_extras = cur.read_u32()? as usize;
    let mut extras = Vec::with_capacity(n_extras.min(cur.remaining()));
    for _ in 0..n_extras {
        let k = read_lp_str(cur)?;
        let v = read_lp_str(cur)?;
        extras.push((k, v));
    }

    Ok(FitConfigBlob {
        max_depth,
        max_k,
        min_samples_leaf,
        reg_alpha,
        path_penalty,
        beam_width,
        transition_delta,
        atom_merge_tolerance,
        top_k,
        teacher_alpha,
        feature_exclusion,
        shape_mode,
        atom_mode,
        cleanup_mode,
        dp_criterion,
        atom_merge_rule,
        extras,
    })
}

fn read_lp_str(cur: &mut Cursor) -> Result<String, String> {
    let len = cur.read_u32()? as usize;
    let bytes = cur.read_bytes(len)?;
    std::str::from_utf8(bytes)
        .map(|s| s.to_string())
        .map_err(|e| format!("bad utf8 in length-prefixed string: {}", e))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// v6 stream codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DagCodec {
    None,
    Deflate,
}

impl DagCodec {
    fn byte(self) -> u8 {
        match self {
            DagCodec::None => CODEC_NONE,
            DagCodec::Deflate => CODEC_DEFLATE,
        }
    }
}

/// The default writer's format version (v6).
pub const DEFAULT_FORMAT_VERSION: u16 = VERSION_V6;

/// Serialize a `FactoredDag` plus its feature metadata and optional
/// fit-time configuration in the default format (v6, DEFLATE streams).
/// v6 records `n_train` as unknown; `serialize_factored_dag_v6` takes it
/// explicitly.
pub fn serialize_factored_dag(
    dag: &FactoredDag,
    feature_names: &[String],
    feature_types: &[FeatureType],
    fit_config: Option<&FitConfigBlob>,
) -> Vec<u8> {
    serialize_factored_dag_v6(
        dag,
        feature_names,
        feature_types,
        fit_config,
        None,
        DagCodec::Deflate,
    )
}

/// `serialize_factored_dag` in an explicit format version (4, 5 or 6).
pub fn serialize_factored_dag_version(
    dag: &FactoredDag,
    feature_names: &[String],
    feature_types: &[FeatureType],
    fit_config: Option<&FitConfigBlob>,
    version: u16,
) -> Result<Vec<u8>, String> {
    match version {
        VERSION_V4 => Ok(serialize_factored_dag_v4(
            dag,
            feature_names,
            feature_types,
            fit_config,
        )),
        VERSION_V5 => Ok(serialize_factored_dag_v5(
            dag,
            feature_names,
            feature_types,
            fit_config,
        )),
        VERSION_V6 => Ok(serialize_factored_dag_v6(
            dag,
            feature_names,
            feature_types,
            fit_config,
            None,
            DagCodec::Deflate,
        )),
        v => Err(format!(
            "cannot write format version {} (supported: 4, 5, 6)",
            v
        )),
    }
}

/// Serialize a `FactoredDag` plus its feature metadata and optional
/// fit-time configuration to v4 bytes (fixed-size records).
///
/// Feature names and types are embedded so that load is robust to a
/// missing `feature_names` argument. Pass empty slices to omit them.
/// `fit_config` lets a saved DAG carry the SolverConfig + Python kwargs it
/// was fit with so a downstream re-fit can reuse the same recipe; pass
/// `None` to omit the section.
pub fn serialize_factored_dag_v4(
    dag: &FactoredDag,
    feature_names: &[String],
    feature_types: &[FeatureType],
    fit_config: Option<&FitConfigBlob>,
) -> Vec<u8> {
    // Phase A — IO format predates inline leaves. Until we bump the
    // wire format (or expand inline leaves into synthetic arena slots
    // on serialize), refuse to serialize a DAG that uses the inline
    // path. Loaded DAGs always have `inline_leaves` empty; only fresh
    // build outputs would land here, and the test runner doesn't
    // currently exercise that case.
    assert!(
        dag.inline_leaves.is_empty() && dag.inline_leaf_costs.is_empty(),
        "serialize_factored_dag: DAG contains {} inline leaves; \
         the v4 wire format predates Phase A inline-leaf storage. \
         Bump the format version (or expand inline leaves into the \
         arena before serializing) to persist this DAG.",
        dag.inline_leaves.len(),
    );
    let dedup = build_dedup(dag);

    let has_group_maps = !dedup.group_maps.is_empty();
    let has_feature_meta = !feature_names.is_empty() || !feature_types.is_empty();
    let has_fit_config = fit_config.is_some();

    let mut flags: u16 = 0;
    if has_group_maps {
        flags |= FLAG_HAS_GROUP_MAPS;
    }
    if has_feature_meta {
        flags |= FLAG_HAS_FEATURE_META;
    }
    if has_fit_config {
        flags |= FLAG_HAS_FIT_CONFIG;
    }
    if dag.keyed_costs {
        flags |= FLAG_KEYED_COSTS;
    }
    if dag.keyed_costs && dag.exact_cost_keys {
        flags |= FLAG_EXACT_COST_KEYS;
    }

    let mut buf: Vec<u8> = Vec::with_capacity(estimate_size(dag));

    // HEADER (32 bytes)
    put_u32(&mut buf, MAGIC_V3);
    put_u16(&mut buf, VERSION_V4);
    put_u16(&mut buf, flags);
    put_u32(&mut buf, dag.subproblems.len() as u32);
    put_u32(&mut buf, dedup.thresholds.len() as u32);
    put_u32(&mut buf, dedup.group_maps.len() as u32);
    put_u32(&mut buf, dag.root_cost_schedule.len() as u32);
    put_u64(&mut buf, 0); // reserved
    debug_assert_eq!(buf.len(), HEADER_LEN);

    // METADATA (64 bytes)
    let md_start = buf.len();
    put_u128(&mut buf, dag.tree_count);
    put_f64(&mut buf, dag.opt);
    put_f64(&mut buf, dag.rashomon_bound);
    put_f64(&mut buf, dag.theta);
    put_f64(&mut buf, dag.reg);
    put_f64(&mut buf, dag.cost_precision);
    put_f32(&mut buf, dag.delta_budget.delta_beam as f32);
    put_f32(&mut buf, dag.delta_budget.delta_compress as f32);
    put_f32(&mut buf, dag.delta_budget.delta_quantize as f32);
    put_f32(&mut buf, dag.delta_budget.delta_criterion as f32);
    debug_assert_eq!(buf.len() - md_start, METADATA_LEN);

    // THRESHOLD_DEDUP
    for arc in &dedup.thresholds {
        put_u32(&mut buf, arc.len() as u32);
        for &t in arc.iter() {
            put_f64(&mut buf, t);
        }
    }

    // GROUP_MAP_DEDUP
    if has_group_maps {
        for arc in &dedup.group_maps {
            put_u32(&mut buf, arc.len() as u32);
            for &g in arc.iter() {
                put_u16(&mut buf, g);
            }
        }
    }

    // SUBPROBLEMS (interleaved fixed record + variable splits chunk)
    for sp in &dag.subproblems {
        write_subproblem(&mut buf, sp, &dedup, dag);
    }

    // ROOT
    put_u32(&mut buf, dag.root_ref);
    for entry in &dag.root_cost_schedule {
        put_f64(&mut buf, entry.cost);
        put_u128(&mut buf, entry.tree_count());
    }

    write_tail(&mut buf, feature_names, feature_types, fit_config);
    buf
}

/// FEATURE_META (when names or types are given), FIT_CONFIG (when given)
/// and the TRAILER (CRC32 of all preceding bytes, then the sentinel).
fn write_tail(
    buf: &mut Vec<u8>,
    feature_names: &[String],
    feature_types: &[FeatureType],
    fit_config: Option<&FitConfigBlob>,
) {
    // FEATURE_META
    if !feature_names.is_empty() || !feature_types.is_empty() {
        let n = feature_names.len().max(feature_types.len()) as u32;
        put_u32(buf, n);
        for i in 0..n as usize {
            let name = feature_names.get(i).cloned().unwrap_or_default();
            let bytes = name.as_bytes();
            put_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(bytes);
            let ft = feature_types.get(i).copied().unwrap_or(FeatureType::Num);
            put_u8(buf, feature_type_to_u8(ft));
        }
    }

    // FIT_CONFIG
    if let Some(fc) = fit_config {
        write_fit_config(buf, fc);
    }

    // TRAILER (8 bytes): CRC32 of all preceding, then sentinel.
    let cksum = crc32(buf);
    put_u32(buf, cksum);
    put_u32(buf, TRAILER_SENTINEL);
}

/// Whether a v5 file may omit the tree counts: the loader's recount
/// (`rashomon_dag::recount_at_bound`) reproduces them only on keyed DAGs
/// whose schedules derive from the splits alone (no build-time
/// `cost_entries`, no prefix / minority leaves, none of which the format
/// persists).
fn v5_counts_omittable(dag: &FactoredDag) -> bool {
    dag.keyed_costs
        && dag.subproblems.iter().all(|sp| {
            sp.cost_entries.is_none() && sp.prefix_leaf.is_none() && sp.leaf_minority.is_none()
        })
}

/// v5 writer (layout in the module docs).
fn serialize_factored_dag_v5(
    dag: &FactoredDag,
    feature_names: &[String],
    feature_types: &[FeatureType],
    fit_config: Option<&FitConfigBlob>,
) -> Vec<u8> {
    assert!(
        dag.inline_leaves.is_empty() && dag.inline_leaf_costs.is_empty(),
        "serialize_factored_dag: DAG contains {} inline leaves; the RSMY \
         formats predate Phase A inline-leaf storage.",
        dag.inline_leaves.len(),
    );
    let counts_stored = !v5_counts_omittable(dag);

    // Content-deduplicated threshold / group_map tables and the spec table
    // (every `spec_pool` entry, in pool order: spec ids are kept).
    let mut thr_ids: HashMap<Vec<u64>, u32> = HashMap::new();
    let mut thr_table: Vec<&Arc<[f64]>> = Vec::new();
    let mut gm_ids: HashMap<Vec<u16>, u32> = HashMap::new();
    let mut gm_table: Vec<&Arc<[u16]>> = Vec::new();
    let mut spec_rows: Vec<(u16, u8, u32, u32)> = Vec::with_capacity(dag.spec_pool.len());
    for spec in &dag.spec_pool {
        let tkey: Vec<u64> = spec.thresholds.iter().map(|t| t.to_bits()).collect();
        let next = thr_table.len() as u32;
        let tid = *thr_ids.entry(tkey).or_insert_with(|| {
            thr_table.push(&spec.thresholds);
            next
        });
        let gid = match &spec.group_map {
            Some(gm) => {
                let next = gm_table.len() as u32;
                let id = *gm_ids.entry(gm.to_vec()).or_insert_with(|| {
                    gm_table.push(gm);
                    next
                });
                id + 1
            }
            None => 0,
        };
        spec_rows.push((spec.feature_index, spec.arity, tid, gid));
    }

    // Split-record table: the promoted pool as is, else the inline splits
    // deduplicated by content (spec id, children, tree_count).
    let promoted = !dag.split_pool.entries.is_empty();
    let mut records: Vec<&FactoredSplit> = Vec::new();
    let mut vertex_splits: Vec<Vec<u32>> = Vec::with_capacity(dag.subproblems.len());
    if promoted {
        records.extend(dag.split_pool.entries.iter());
        for sp in &dag.subproblems {
            vertex_splits.push(if sp.split_ids.is_empty() {
                Vec::new()
            } else {
                sp.split_ids.clone()
            });
        }
        debug_assert!(dag.subproblems.iter().all(|sp| sp.splits.is_empty()));
    } else {
        let mut ids: HashMap<(u32, Vec<FSubRef>, u128), u32> = HashMap::new();
        for sp in &dag.subproblems {
            let mut v = Vec::with_capacity(sp.splits.len());
            for split in &sp.splits {
                let key = (split.spec_id, split.children.to_vec(), split.tree_count());
                let next = records.len() as u32;
                let id = *ids.entry(key).or_insert_with(|| {
                    records.push(split);
                    next
                });
                v.push(id);
            }
            vertex_splits.push(v);
        }
    }

    // Leaf-value table: distinct (cost, loss) bit pairs.
    let mut leaf_ids: HashMap<(u64, u64), u32> = HashMap::new();
    let mut leaf_values: Vec<(f64, f64)> = Vec::new();
    let vertex_leaf: Vec<Option<u32>> = dag
        .subproblems
        .iter()
        .map(|sp| {
            sp.leaf.as_ref().map(|(c, info)| {
                let next = leaf_values.len() as u32;
                *leaf_ids
                    .entry((c.to_bits(), info.loss.to_bits()))
                    .or_insert_with(|| {
                        leaf_values.push((*c, info.loss));
                        next
                    })
            })
        })
        .collect();

    let mut flags: u16 = 0;
    if !gm_table.is_empty() {
        flags |= FLAG_HAS_GROUP_MAPS;
    }
    if !feature_names.is_empty() || !feature_types.is_empty() {
        flags |= FLAG_HAS_FEATURE_META;
    }
    if fit_config.is_some() {
        flags |= FLAG_HAS_FIT_CONFIG;
    }
    if dag.keyed_costs {
        flags |= FLAG_KEYED_COSTS;
    }
    if dag.keyed_costs && dag.exact_cost_keys {
        flags |= FLAG_EXACT_COST_KEYS;
    }
    if counts_stored {
        flags |= FLAG_COUNTS_STORED;
    }

    let mut buf: Vec<u8> = Vec::new();
    // HEADER (32 bytes; v4 layout, reserved word = n_specs, n_split_records)
    put_u32(&mut buf, MAGIC_V3);
    put_u16(&mut buf, VERSION_V5);
    put_u16(&mut buf, flags);
    put_u32(&mut buf, dag.subproblems.len() as u32);
    put_u32(&mut buf, thr_table.len() as u32);
    put_u32(&mut buf, gm_table.len() as u32);
    put_u32(
        &mut buf,
        if counts_stored {
            dag.root_cost_schedule.len() as u32
        } else {
            0
        },
    );
    put_u32(&mut buf, spec_rows.len() as u32);
    put_u32(&mut buf, records.len() as u32);
    debug_assert_eq!(buf.len(), HEADER_LEN);

    // METADATA (v4 layout; tree_count is the load-time recount check)
    let md_start = buf.len();
    put_u128(&mut buf, dag.tree_count);
    put_f64(&mut buf, dag.opt);
    put_f64(&mut buf, dag.rashomon_bound);
    put_f64(&mut buf, dag.theta);
    put_f64(&mut buf, dag.reg);
    put_f64(&mut buf, dag.cost_precision);
    put_f32(&mut buf, dag.delta_budget.delta_beam as f32);
    put_f32(&mut buf, dag.delta_budget.delta_compress as f32);
    put_f32(&mut buf, dag.delta_budget.delta_quantize as f32);
    put_f32(&mut buf, dag.delta_budget.delta_criterion as f32);
    debug_assert_eq!(buf.len() - md_start, METADATA_LEN);

    // THRESHOLDS: varint length, raw f64s.
    for arc in &thr_table {
        put_varint(&mut buf, arc.len() as u128);
        for &t in arc.iter() {
            put_f64(&mut buf, t);
        }
    }
    // GROUP_MAPS: varint length, varint entries.
    for arc in &gm_table {
        put_varint(&mut buf, arc.len() as u128);
        for &g in arc.iter() {
            put_varint(&mut buf, g as u128);
        }
    }
    // SPECS: feature, arity, threshold id, group_map id + 1 (0 = none).
    for &(feat, arity, tid, gid) in &spec_rows {
        put_varint(&mut buf, feat as u128);
        put_u8(&mut buf, arity);
        put_varint(&mut buf, tid as u128);
        put_varint(&mut buf, gid as u128);
    }
    // LEAF VALUES: count, then (cost, loss) f64 pairs.
    put_varint(&mut buf, leaf_values.len() as u128);
    for &(c, l) in &leaf_values {
        put_f64(&mut buf, c);
        put_f64(&mut buf, l);
    }
    // SPLIT RECORDS: spec id, `arity` child vertex ids [, tree_count].
    for split in &records {
        let arity = dag.spec_pool[split.spec_id as usize].arity as usize;
        assert_eq!(
            split.children.len(),
            arity,
            "v5: split children != spec arity"
        );
        put_varint(&mut buf, split.spec_id as u128);
        for &c in split.children.iter() {
            put_varint(&mut buf, c as u128);
        }
        if counts_stored {
            put_varint(&mut buf, split.tree_count());
        }
    }
    // VERTICES: flags (bit 0 leaf) [, prediction, leaf-value id, n_samples],
    // n_splits, split-record ids [, tree_count].
    for (i, sp) in dag.subproblems.iter().enumerate() {
        match (&sp.leaf, vertex_leaf[i]) {
            (Some((_, info)), Some(lid)) => {
                put_u8(&mut buf, 1);
                put_u8(&mut buf, info.prediction);
                put_varint(&mut buf, lid as u128);
                put_varint(&mut buf, info.n_samples as u128);
            }
            _ => put_u8(&mut buf, 0),
        }
        put_varint(&mut buf, vertex_splits[i].len() as u128);
        for &id in &vertex_splits[i] {
            put_varint(&mut buf, id as u128);
        }
        if counts_stored {
            put_varint(&mut buf, sp.tree_count);
        }
    }
    // ROOT: root ref [, root schedule (cost f64, varint count)].
    put_varint(&mut buf, dag.root_ref as u128);
    if counts_stored {
        for entry in &dag.root_cost_schedule {
            put_f64(&mut buf, entry.cost);
            put_varint(&mut buf, entry.tree_count());
        }
    }

    write_tail(&mut buf, feature_names, feature_types, fit_config);
    buf
}

/// v6: one compressed stream (`varint raw_len, varint comp_len, bytes`).
fn put_stream(buf: &mut Vec<u8>, codec: DagCodec, raw: &[u8]) {
    put_varint(buf, raw.len() as u128);
    match codec {
        DagCodec::None => {
            put_varint(buf, raw.len() as u128);
            buf.extend_from_slice(raw);
        }
        DagCodec::Deflate => {
            use std::io::Write;
            let mut enc =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::best());
            enc.write_all(raw).expect("deflate into a Vec cannot fail");
            let comp = enc.finish().expect("deflate into a Vec cannot fail");
            put_varint(buf, comp.len() as u128);
            buf.extend_from_slice(&comp);
        }
    }
}

/// v6 integer leaf code: `k` with `(k / denom, k / denom + leaf_reg)` equal
/// bit for bit to `(loss, cost)`, else None (the leaf is escaped).
fn v6_leaf_k(cost: f64, loss: f64, leaf_reg: f64, denom: u64) -> Option<u64> {
    const EXACT: f64 = 9_007_199_254_740_992.0; // 2^53
    if denom == 0 || denom as f64 >= EXACT {
        return None;
    }
    let d = denom as f64;
    let kf = (loss * d).round();
    if !(kf >= 0.0 && kf < EXACT) {
        return None;
    }
    let k = kf as u64;
    let l = (k as f64) / d;
    (l.to_bits() == loss.to_bits() && (l + leaf_reg).to_bits() == cost.to_bits()).then_some(k)
}

/// v6 leaf constants `(leaf_reg, leaf_denom)`. leaf_reg candidates are
/// `dag.reg` and cost - loss of the first leaf (the latter can differ from
/// the build's reg by an ulp, which would escape most other leaves); for
/// each, leaf_denom = the first candidate that makes every distinct leaf
/// value exact (caller `n_train`, root leaf n_samples, max leaf n_samples,
/// then denominators of the largest-support leaf's loss), else the one with
/// the fewest escapes (0 = all escaped). The pair with the fewest escapes
/// wins (ties: `dag.reg`).
fn v6_leaf_constants(
    dag: &FactoredDag,
    leaf_order: &[u32],
    internal_order: &[u32],
    n_train: Option<u64>,
) -> (f64, u64) {
    let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    let mut values: Vec<(f64, f64, usize)> = Vec::new();
    for &v in leaf_order.iter().chain(internal_order) {
        if let Some((c, info)) = &dag.subproblems[v as usize].leaf {
            if seen.insert((c.to_bits(), info.loss.to_bits())) {
                values.push((*c, info.loss, info.n_samples));
            }
        }
    }
    let Some(&(c0, l0, _)) = values.first() else {
        return (dag.reg, 0);
    };
    let root_n = dag
        .subproblems
        .get(dag.root_ref as usize)
        .and_then(|sp| sp.leaf.as_ref())
        .map(|(_, info)| info.n_samples as u64);
    let max_n = values.iter().map(|&(_, _, n)| n as u64).max();
    let widest = values
        .iter()
        .filter(|&&(_, l, _)| l > 0.0 && l.is_finite())
        .max_by_key(|&&(_, _, n)| n)
        .map(|&(_, l, s)| (l, s));
    // (escapes, leaf_denom) for one leaf_reg.
    let denom_for = |leaf_reg: f64| -> (usize, u64) {
        let failures = |d: u64| {
            values
                .iter()
                .filter(|&&(c, l, _)| v6_leaf_k(c, l, leaf_reg, d).is_none())
                .count()
        };
        let mut tried: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut best = (values.len(), 0u64);
        let mut consider = |d: u64, best: &mut (usize, u64)| -> bool {
            if d == 0 || !tried.insert(d) {
                return false;
            }
            let f = failures(d);
            if f < best.0 {
                *best = (f, d);
            }
            f == 0
        };
        for d in [n_train, root_n, max_n].into_iter().flatten() {
            if consider(d, &mut best) {
                return best;
            }
        }
        if let Some((loss, s)) = widest {
            for e in 1..=(s as u64) {
                let df = (e as f64 / loss).round();
                if !(df >= 1.0 && df < 9_007_199_254_740_992.0) {
                    continue;
                }
                if ((e as f64) / df).to_bits() == loss.to_bits() && consider(df as u64, &mut best) {
                    return best;
                }
            }
        }
        best
    };
    let mut best: Option<(usize, f64, u64)> = None;
    for leaf_reg in [dag.reg, c0 - l0] {
        if best.is_some_and(|(f, r, _)| f == 0 || r.to_bits() == leaf_reg.to_bits()) {
            continue;
        }
        let (f, d) = denom_for(leaf_reg);
        if best.map_or(true, |(bf, _, _)| f < bf) {
            best = Some((f, leaf_reg, d));
        }
    }
    let (_, leaf_reg, leaf_denom) = best.unwrap();
    (leaf_reg, leaf_denom)
}

/// v6 internal-vertex order: DFS post-order from the root over every
/// split's non-leaf-only children, then the unreachable internal vertices
/// (post-order sweep over the remaining ids). Every internal child precedes
/// its parent.
fn v6_internal_post_order(dag: &FactoredDag, leaf_only: &[bool]) -> Vec<u32> {
    let n = dag.subproblems.len();
    let mut state = vec![0u8; n];
    let mut post: Vec<u32> = Vec::with_capacity(n);
    let starts = std::iter::once(dag.root_ref as usize).chain(0..n);
    for start in starts {
        if start >= n || leaf_only[start] || state[start] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, bool)> = vec![(start, false)];
        while let Some((i, children_done)) = stack.pop() {
            if children_done {
                state[i] = 2;
                post.push(i as u32);
                continue;
            }
            if state[i] != 0 {
                continue;
            }
            state[i] = 1;
            stack.push((i, true));
            let children: Vec<usize> = subproblem_splits(dag, &dag.subproblems[i])
                .flat_map(|split| split.children.iter().map(|&c| c as usize))
                .filter(|&c| !leaf_only[c] && state[c] == 0)
                .collect();
            for c in children.into_iter().rev() {
                stack.push((c, false));
            }
        }
    }
    post
}

/// v6 writer (layout in the module docs). `n_train` is
/// the caller's training-row count (recorded, and the first leaf
/// denominator tried); `codec` compresses each stream.
pub fn serialize_factored_dag_v6(
    dag: &FactoredDag,
    feature_names: &[String],
    feature_types: &[FeatureType],
    fit_config: Option<&FitConfigBlob>,
    n_train: Option<u64>,
    codec: DagCodec,
) -> Vec<u8> {
    assert!(
        dag.inline_leaves.is_empty() && dag.inline_leaf_costs.is_empty(),
        "serialize_factored_dag: DAG contains {} inline leaves; the RSMY \
         formats predate Phase A inline-leaf storage.",
        dag.inline_leaves.len(),
    );
    let counts_stored = !v5_counts_omittable(dag);
    let n = dag.subproblems.len();

    // Counting pass: split counts, child references, spec uses.
    let mut n_splits = vec![0u32; n];
    let mut ref_count = vec![0u32; n];
    let mut spec_uses = vec![0u64; dag.spec_pool.len()];
    for (v, sp) in dag.subproblems.iter().enumerate() {
        for split in subproblem_splits(dag, sp) {
            n_splits[v] += 1;
            spec_uses[split.spec_id as usize] += 1;
            for &c in split.children.iter() {
                assert!((c as usize) < n, "v6: child ref {} out of bounds", c);
                ref_count[c as usize] += 1;
            }
        }
    }
    let leaf_only: Vec<bool> = (0..n)
        .map(|v| dag.subproblems[v].leaf.is_some() && n_splits[v] == 0)
        .collect();

    // Orders and id maps.
    let mut leaf_order: Vec<u32> = (0..n as u32).filter(|&v| leaf_only[v as usize]).collect();
    leaf_order.sort_by(|&a, &b| {
        ref_count[b as usize]
            .cmp(&ref_count[a as usize])
            .then(a.cmp(&b))
    });
    let internal_order = v6_internal_post_order(dag, &leaf_only);
    let n_leaf = leaf_order.len();
    debug_assert_eq!(n_leaf + internal_order.len(), n);
    let mut new_id = vec![u32::MAX; n];
    for (i, &v) in leaf_order.iter().enumerate() {
        new_id[v as usize] = i as u32;
    }
    for (p, &v) in internal_order.iter().enumerate() {
        new_id[v as usize] = (n_leaf + p) as u32;
    }
    let mut spec_order: Vec<u32> = (0..dag.spec_pool.len() as u32)
        .filter(|&s| spec_uses[s as usize] > 0)
        .collect();
    spec_order.sort_by(|&a, &b| {
        spec_uses[b as usize]
            .cmp(&spec_uses[a as usize])
            .then(a.cmp(&b))
    });
    let mut spec_rank = vec![u32::MAX; dag.spec_pool.len()];
    for (r, &s) in spec_order.iter().enumerate() {
        spec_rank[s as usize] = r as u32;
    }

    // Threshold grid per feature (ascending, dedup by bits) and group maps
    // (content-deduplicated, first use in rank order).
    let n_grid = spec_order
        .iter()
        .map(|&s| dag.spec_pool[s as usize].feature_index as usize + 1)
        .max()
        .unwrap_or(0);
    let mut grid: Vec<Vec<f64>> = vec![Vec::new(); n_grid];
    for &s in &spec_order {
        let spec = &dag.spec_pool[s as usize];
        grid[spec.feature_index as usize].extend(spec.thresholds.iter().copied());
    }
    let mut grid_idx: Vec<HashMap<u64, u32>> = Vec::with_capacity(n_grid);
    for g in grid.iter_mut() {
        g.sort_by(|a, b| a.total_cmp(b));
        g.dedup_by(|a, b| a.to_bits() == b.to_bits());
        grid_idx.push(
            g.iter()
                .enumerate()
                .map(|(i, t)| (t.to_bits(), i as u32))
                .collect(),
        );
    }
    // The SPECS gap code needs each spec's thresholds strictly ascending in
    // grid order; otherwise fall back silently to v5 (lossless for any
    // thresholds).
    let ascending = spec_order.iter().all(|&s| {
        let spec = &dag.spec_pool[s as usize];
        let ix = &grid_idx[spec.feature_index as usize];
        spec.thresholds
            .windows(2)
            .all(|w| ix[&w[0].to_bits()] < ix[&w[1].to_bits()])
    });
    if !ascending {
        return serialize_factored_dag_v5(dag, feature_names, feature_types, fit_config);
    }
    let mut gm_ids: HashMap<Vec<u16>, u32> = HashMap::new();
    let mut gm_table: Vec<&Arc<[u16]>> = Vec::new();

    let mut s_grid = Vec::new();
    put_varint(&mut s_grid, n_grid as u128);
    for g in &grid {
        put_varint(&mut s_grid, g.len() as u128);
        for &t in g {
            put_f64(&mut s_grid, t);
        }
    }
    let mut s_specs = Vec::new();
    for &s in &spec_order {
        let spec = &dag.spec_pool[s as usize];
        let f = spec.feature_index as usize;
        put_varint(&mut s_specs, f as u128);
        put_u8(&mut s_specs, spec.arity);
        put_varint(&mut s_specs, spec.thresholds.len() as u128);
        let mut prev: i64 = -1;
        for t in spec.thresholds.iter() {
            let idx = grid_idx[f][&t.to_bits()] as i64;
            debug_assert!(idx > prev, "checked by the ascending pre-pass above");
            put_varint(&mut s_specs, (idx - prev - 1) as u128);
            prev = idx;
        }
        let gid = match &spec.group_map {
            Some(gm) => {
                let next = gm_table.len() as u32;
                let id = *gm_ids.entry(gm.to_vec()).or_insert_with(|| {
                    gm_table.push(gm);
                    next
                });
                id + 1
            }
            None => 0,
        };
        put_varint(&mut s_specs, gid as u128);
    }
    let mut s_gmaps = Vec::new();
    if !gm_table.is_empty() {
        put_varint(&mut s_gmaps, gm_table.len() as u128);
        for gm in &gm_table {
            put_varint(&mut s_gmaps, gm.len() as u128);
            for &g in gm.iter() {
                put_varint(&mut s_gmaps, g as u128);
            }
        }
    }

    // Leaf options (leaf-only vertices, then internal vertices with a leaf).
    let (leaf_reg, leaf_denom) = v6_leaf_constants(dag, &leaf_order, &internal_order, n_train);
    let (mut s_pred, mut s_kcode, mut s_kesc, mut s_nsamp) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_esc: u64 = 0;
    let mut put_leaf = |leaf: &(f64, RLeafInfo)| {
        let (cost, info) = leaf;
        put_u8(&mut s_pred, info.prediction);
        match v6_leaf_k(*cost, info.loss, leaf_reg, leaf_denom) {
            Some(k) => put_varint(&mut s_kcode, 2 * k as u128),
            None => {
                put_varint(&mut s_kcode, 2 * n_esc as u128 + 1);
                put_f64(&mut s_kesc, *cost);
                put_f64(&mut s_kesc, info.loss);
                n_esc += 1;
            }
        }
        put_varint(&mut s_nsamp, info.n_samples as u128);
    };
    for &v in &leaf_order {
        put_leaf(dag.subproblems[v as usize].leaf.as_ref().unwrap());
    }
    for &v in &internal_order {
        if let Some(leaf) = &dag.subproblems[v as usize].leaf {
            put_leaf(leaf);
        }
    }

    // Internal vertices: header, splits (spec rank, child refs) [, counts].
    let (mut s_vhdr, mut s_specids, mut s_tags, mut s_lref, mut s_iref, mut s_counts) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut n_tag_bits: usize = 0;
    for (pos, &v) in internal_order.iter().enumerate() {
        let sp = &dag.subproblems[v as usize];
        put_varint(
            &mut s_vhdr,
            2 * n_splits[v as usize] as u128 + sp.leaf.is_some() as u128,
        );
        for split in subproblem_splits(dag, sp) {
            let arity = dag.spec_pool[split.spec_id as usize].arity as usize;
            assert_eq!(
                split.children.len(),
                arity,
                "v6: split children != spec arity"
            );
            put_varint(&mut s_specids, spec_rank[split.spec_id as usize] as u128);
            for &c in split.children.iter() {
                if n_tag_bits % 8 == 0 {
                    s_tags.push(0);
                }
                if leaf_only[c as usize] {
                    *s_tags.last_mut().unwrap() |= 1 << (n_tag_bits % 8);
                    put_varint(&mut s_lref, new_id[c as usize] as u128);
                } else {
                    let child_pos = new_id[c as usize] as usize - n_leaf;
                    assert!(child_pos < pos, "v6: internal child not before its parent");
                    put_varint(&mut s_iref, (pos - child_pos) as u128);
                }
                n_tag_bits += 1;
            }
            if counts_stored {
                put_varint(&mut s_counts, split.tree_count());
            }
        }
    }
    if counts_stored {
        for &v in leaf_order.iter().chain(&internal_order) {
            put_varint(&mut s_counts, dag.subproblems[v as usize].tree_count);
        }
        for entry in &dag.root_cost_schedule {
            put_f64(&mut s_counts, entry.cost);
            put_varint(&mut s_counts, entry.tree_count());
        }
    }

    let mut flags: u16 = 0;
    if !gm_table.is_empty() {
        flags |= FLAG_HAS_GROUP_MAPS;
    }
    if !feature_names.is_empty() || !feature_types.is_empty() {
        flags |= FLAG_HAS_FEATURE_META;
    }
    if fit_config.is_some() {
        flags |= FLAG_HAS_FIT_CONFIG;
    }
    if dag.keyed_costs {
        flags |= FLAG_KEYED_COSTS;
    }
    if dag.keyed_costs && dag.exact_cost_keys {
        flags |= FLAG_EXACT_COST_KEYS;
    }
    if counts_stored {
        flags |= FLAG_COUNTS_STORED;
    }

    let mut buf: Vec<u8> = Vec::new();
    // HEADER (32 bytes)
    put_u32(&mut buf, MAGIC_V3);
    put_u16(&mut buf, VERSION_V6);
    put_u16(&mut buf, flags);
    put_u32(&mut buf, n as u32);
    put_u32(&mut buf, n_leaf as u32);
    put_u32(&mut buf, internal_order.len() as u32);
    put_u32(
        &mut buf,
        if counts_stored {
            dag.root_cost_schedule.len() as u32
        } else {
            0
        },
    );
    put_u8(&mut buf, codec.byte());
    put_u8(&mut buf, V6_N_STREAMS + counts_stored as u8);
    put_u16(&mut buf, 0);
    put_u32(&mut buf, spec_order.len() as u32);
    debug_assert_eq!(buf.len(), HEADER_LEN);

    // METADATA (v4/v5 layout)
    let md_start = buf.len();
    put_u128(&mut buf, dag.tree_count);
    put_f64(&mut buf, dag.opt);
    put_f64(&mut buf, dag.rashomon_bound);
    put_f64(&mut buf, dag.theta);
    put_f64(&mut buf, dag.reg);
    put_f64(&mut buf, dag.cost_precision);
    put_f32(&mut buf, dag.delta_budget.delta_beam as f32);
    put_f32(&mut buf, dag.delta_budget.delta_compress as f32);
    put_f32(&mut buf, dag.delta_budget.delta_quantize as f32);
    put_f32(&mut buf, dag.delta_budget.delta_criterion as f32);
    debug_assert_eq!(buf.len() - md_start, METADATA_LEN);

    // METADATA_V6
    put_f64(&mut buf, leaf_reg);
    put_varint(&mut buf, leaf_denom as u128);
    put_varint(&mut buf, n_train.unwrap_or(0) as u128);
    put_varint(&mut buf, new_id[dag.root_ref as usize] as u128);
    put_varint(&mut buf, n_esc as u128);

    // STREAMS
    for raw in [
        &s_grid, &s_gmaps, &s_specs, &s_vhdr, &s_pred, &s_kcode, &s_kesc, &s_nsamp, &s_specids,
        &s_tags, &s_lref, &s_iref,
    ] {
        put_stream(&mut buf, codec, raw);
    }
    if counts_stored {
        put_stream(&mut buf, codec, &s_counts);
    }

    write_tail(&mut buf, feature_names, feature_types, fit_config);
    buf
}

fn write_subproblem(
    buf: &mut Vec<u8>,
    sp: &FactoredSubproblem,
    dedup: &DedupTables,
    dag: &FactoredDag,
) {
    let (has_leaf, prediction, leaf_cost, leaf_loss, leaf_n) = match &sp.leaf {
        Some((cost, info)) => (
            true,
            info.prediction,
            *cost,
            info.loss,
            info.n_samples as u64,
        ),
        None => (false, 0u8, 0.0, 0.0, 0u64),
    };
    put_u8(buf, if has_leaf { 1 } else { 0 });
    put_u8(buf, prediction);
    put_u16(buf, subproblem_n_splits(sp) as u16);
    put_f64(buf, leaf_cost);
    put_f64(buf, leaf_loss);
    put_u64(buf, leaf_n);
    put_u128(buf, sp.tree_count);

    for split in subproblem_splits(dag, sp) {
        write_split(buf, split, dedup, dag);
    }
}

fn write_split(buf: &mut Vec<u8>, split: &FactoredSplit, dedup: &DedupTables, dag: &FactoredDag) {
    // Resolve the canonical spec_id to its SplitSpec. The on-disk
    // format is unchanged: feature/arity/threshold_id/group_map_id are still
    // written inline per split, so older readers and the v3/v4 format are
    // untouched (the spec_pool is rebuilt on load).
    let spec = &dag.spec_pool[split.spec_id as usize];
    let key = Arc::as_ptr(&spec.thresholds) as *const () as usize;
    let threshold_id = *dedup.threshold_id.get(&key).expect("threshold interned");
    let group_map_id = match &spec.group_map {
        Some(gm) => {
            let k = Arc::as_ptr(gm) as *const () as usize;
            *dedup.group_map_id.get(&k).expect("group_map interned")
        }
        None => NO_GROUP_MAP,
    };

    put_u16(buf, spec.feature_index);
    put_u8(buf, spec.arity);
    put_u8(buf, 0); // pad
    put_u32(buf, threshold_id);
    put_u32(buf, group_map_id);
    // On-disk count is a u128 (the in-memory logical width).
    put_u128(buf, split.tree_count());

    // v4+: per-edge cost schedules are not stored. Each child is
    // serialized as just its FSubRef; schedules are recomputed at
    // extract time via `materialize_schedule`.
    debug_assert_eq!(split.children.len(), spec.arity as usize);
    for fsub_ref in &split.children {
        put_u32(buf, *fsub_ref);
    }
}

fn estimate_size(dag: &FactoredDag) -> usize {
    // Coarse upper bound. Per-edge cost schedules are not stored;
    // account only for the per-subproblem fixed overhead
    // plus child FSubRef slots and the root schedule.
    // Walk via the pool when promoted; the per-subproblem split
    // count via subproblem_n_splits dispatches on split_ids vs splits.
    let total_children: usize = dag
        .subproblems
        .iter()
        .flat_map(|sp| subproblem_splits(dag, sp))
        .map(|sp| sp.children.len())
        .sum();
    HEADER_LEN
        + METADATA_LEN
        + dag.subproblems.len() * 64
        + total_children * 4
        + dag.root_cost_schedule.len() * 24
        + 256
}

// ---------------------------------------------------------------------------
// Deserialize
// ---------------------------------------------------------------------------

/// A deserialized file: the DAG, its feature metadata, the optional fit
/// configuration and the recorded training-row count (v6 only; None for
/// v4 / v5 files and for v6 files written without one).
pub struct LoadedDag {
    pub dag: FactoredDag,
    pub feature_names: Vec<String>,
    pub feature_types: Vec<FeatureType>,
    pub fit_config: Option<FitConfigBlob>,
    pub n_train: Option<u64>,
}

/// Deserialize a v4 / v5 / v6 byte stream into a `FactoredDag`, feature
/// metadata, and optional fit configuration.
///
/// Returns `(dag, feature_names, feature_types, fit_config)`. The trailing
/// CRC32 is verified before any structural parsing returns the result, so
/// a corrupted file fails fast with a clear message. Any missing optional
/// section comes back as an empty vec / `None`.
pub fn deserialize_factored_dag(
    data: &[u8],
) -> Result<
    (
        FactoredDag,
        Vec<String>,
        Vec<FeatureType>,
        Option<FitConfigBlob>,
    ),
    String,
> {
    let l = deserialize_factored_dag_full(data)?;
    Ok((l.dag, l.feature_names, l.feature_types, l.fit_config))
}

/// `deserialize_factored_dag` that also returns the recorded `n_train`.
pub fn deserialize_factored_dag_full(data: &[u8]) -> Result<LoadedDag, String> {
    if data.len() < HEADER_LEN + METADATA_LEN + 8 {
        return Err(format!(
            "input too short for an RSMY DAG (got {} bytes, need at least {})",
            data.len(),
            HEADER_LEN + METADATA_LEN + 8
        ));
    }

    // Trailer first: validate before we trust any sizes from the header.
    let trailer_off = data.len() - 8;
    let stored_crc = u32::from_le_bytes(data[trailer_off..trailer_off + 4].try_into().unwrap());
    let sentinel = u32::from_le_bytes(data[trailer_off + 4..trailer_off + 8].try_into().unwrap());
    if sentinel != TRAILER_SENTINEL {
        return Err(format!(
            "missing trailer sentinel: got 0x{:08X}, expected 0x{:08X}",
            sentinel, TRAILER_SENTINEL
        ));
    }
    let computed_crc = crc32(&data[..trailer_off]);
    if computed_crc != stored_crc {
        return Err(format!(
            "CRC32 mismatch: stored 0x{:08X}, computed 0x{:08X}",
            stored_crc, computed_crc
        ));
    }

    let mut cur = Cursor::new(&data[..trailer_off]);

    // HEADER
    let magic = cur.read_u32()?;
    if magic != MAGIC_V3 {
        return Err(format!(
            "wrong magic: got 0x{:08X}, expected 0x{:08X} (RSMY); this build reads RSMY formats \
             v4, v5 and v6 (v2 RSMX files are not supported)",
            magic, MAGIC_V3
        ));
    }
    let version = cur.read_u16()?;
    if version == VERSION_V3 {
        return Err(
            "incompatible version 3, this build reads formats v4-v6. v3 files \
             store per-edge schedules that are no longer supported. \
             Regenerate the DAG (re-fit) and save it again."
                .to_string(),
        );
    }
    if version != VERSION_V4 && version != VERSION_V5 && version != VERSION_V6 {
        return Err(format!(
            "incompatible version {}, this build reads v{}, v{} and v{}",
            version, VERSION_V4, VERSION_V5, VERSION_V6
        ));
    }
    let flags = cur.read_u16()?;
    let n_subproblems = cur.read_u32()? as usize;
    let counts_stored = version == VERSION_V4 || (flags & FLAG_COUNTS_STORED) != 0;
    let (n_thresholds, n_group_maps, n_root_entries, n_specs, n_split_records, v6_header) =
        if version == VERSION_V6 {
            // v6: n_leaf_only, n_internal, n_root_entries, codec u8,
            // n_streams u8, reserved u16, n_specs.
            let n_leaf_only = cur.read_u32()? as usize;
            let n_internal = cur.read_u32()? as usize;
            let n_root_entries = cur.read_u32()? as usize;
            let codec = cur.read_u8()?;
            let n_streams = cur.read_u8()?;
            let _reserved = cur.read_u16()?;
            let n_specs = cur.read_u32()? as usize;
            let hdr = V6Header {
                n_subproblems,
                n_leaf_only,
                n_internal,
                n_root_entries,
                codec,
                n_streams,
                n_specs,
            };
            (0, 0, n_root_entries, n_specs, 0, Some(hdr))
        } else {
            let n_thresholds = cur.read_u32()? as usize;
            let n_group_maps = cur.read_u32()? as usize;
            let n_root_entries = cur.read_u32()? as usize;
            // v4: reserved u64; v5: n_specs, n_split_records.
            let n_specs = cur.read_u32()? as usize;
            let n_split_records = cur.read_u32()? as usize;
            (
                n_thresholds,
                n_group_maps,
                n_root_entries,
                n_specs,
                n_split_records,
                None,
            )
        };

    let has_group_maps = (flags & FLAG_HAS_GROUP_MAPS) != 0;
    let has_feature_meta = (flags & FLAG_HAS_FEATURE_META) != 0;
    let has_fit_config = (flags & FLAG_HAS_FIT_CONFIG) != 0;
    let keyed_costs = (flags & FLAG_KEYED_COSTS) != 0;
    let exact_cost_keys = keyed_costs && (flags & FLAG_EXACT_COST_KEYS) != 0;
    if !has_group_maps && n_group_maps != 0 {
        return Err("group_map dedup table count nonzero but flag clear".into());
    }

    // METADATA
    let tree_count = cur.read_u128()?;
    let opt = cur.read_f64()?;
    let rashomon_bound = cur.read_f64()?;
    let theta = cur.read_f64()?;
    let reg = cur.read_f64()?;
    let cost_precision = cur.read_f64()?;
    let delta_beam = cur.read_f32()? as f64;
    let delta_compress = cur.read_f32()? as f64;
    let delta_quantize = cur.read_f32()? as f64;
    let delta_criterion = cur.read_f32()? as f64;
    let delta_total = delta_beam + delta_compress + delta_quantize + delta_criterion;

    let mut n_train: Option<u64> = None;
    let (subproblems, spec_pool_vec, root_ref, root_cost_schedule) = if let Some(hdr) = &v6_header {
        let md6 = MetadataV6 {
            leaf_reg: cur.read_f64()?,
            leaf_denom: cur.read_varint_u64()?,
            n_train: cur.read_varint_u64()?,
            root_ref: cur.read_varint_u32()?,
            n_leaf_escapes: cur.read_varint_u64()?,
        };
        n_train = (md6.n_train > 0).then_some(md6.n_train);
        read_body_v6(&mut cur, hdr, &md6, has_group_maps, counts_stored)?
    } else if version == VERSION_V5 {
        read_body_v5(
            &mut cur,
            V5Counts {
                n_subproblems,
                n_thresholds,
                n_group_maps,
                n_root_entries,
                n_specs,
                n_split_records,
            },
            counts_stored,
        )?
    } else {
        read_body_v4(
            &mut cur,
            n_subproblems,
            n_thresholds,
            n_group_maps,
            n_root_entries,
            has_group_maps,
        )?
    };

    // FEATURE_META
    let mut feature_names: Vec<String> = Vec::new();
    let mut feature_types: Vec<FeatureType> = Vec::new();
    if has_feature_meta {
        let n_features = cur.read_u32()? as usize;
        feature_names.reserve(n_features);
        feature_types.reserve(n_features);
        for _ in 0..n_features {
            let name_len = cur.read_u32()? as usize;
            let name_bytes = cur.read_bytes(name_len)?;
            let name = std::str::from_utf8(name_bytes)
                .map_err(|e| format!("bad feature name utf8: {}", e))?
                .to_string();
            let ft = u8_to_feature_type(cur.read_u8()?)?;
            feature_names.push(name);
            feature_types.push(ft);
        }
    }

    // FIT_CONFIG (optional)
    let fit_config = if has_fit_config {
        Some(read_fit_config(&mut cur)?)
    } else {
        None
    };

    // Recompute `min_obj` bottom-up (post-order, any arena
    // order) before constructing the DAG, so downstream calls to
    // `materialize_schedule` short-circuit correctly.
    let mut subproblems = subproblems;
    if counts_stored {
        recompute_min_obj_arena(&mut subproblems);
    }

    let mut dag = FactoredDag {
        subproblems,
        root_ref,
        root_cost_schedule,
        tree_count,
        opt,
        rashomon_bound,
        theta,
        reg,
        delta_budget: DeltaBudget {
            delta_beam,
            delta_compress,
            delta_quantize,
            delta_criterion,
            delta_total,
            ..DeltaBudget::default()
        },
        cost_precision,
        // Header bit; files predating it load with float comparisons.
        keyed_costs,
        exact_cost_keys,
        // v3 format predates prefix-leaf metadata. Phase A leaves these
        // fields default-empty on load; a future format bump will persist them
        // when Phase B starts producing populated values.
        // Pool telemetry is fit-time-only; not persisted in v3 format.
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
        // Empty schedule cache; populated lazily on extract.
        schedule_cache: std::sync::Arc::new(dashmap::DashMap::new()),
        wide_schedule_cache: Default::default(),
        // Loaded DAGs keep splits inline (preserves byte-exact
        // roundtrip semantics for the integration test fixtures). Readers
        // dispatch via `rashomon_dag::iter_splits` which handles both
        // inline and pooled storage transparently.
        split_pool: crate::types::FactoredSplitPool::new(),
        // Rebuilt spec pool (deduplicated on load).
        spec_pool: spec_pool_vec,
        // Phase A: on-disk format predates inline leaves (synthetic leaf
        // subproblems are expanded back into the arena at serialize time),
        // so loaded DAGs start with empty inline leaf storage.
        inline_leaves: Vec::new(),
        inline_leaf_costs: Vec::new(),
        inline_leaves_minority: Vec::new(),
        inline_leaf_minority_costs: Vec::new(),
        n_quotient_hash_collisions: 0,
        count_saturated: false,
    };
    if !counts_stored {
        // v5 / v6 without counts: the build's count pass at the stored beta.
        crate::rashomon_dag::recount_at_bound(&mut dag);
        if dag.tree_count != tree_count {
            return Err(format!(
                "v{} recount gives tree_count {} but the file records {}",
                version, dag.tree_count, tree_count
            ));
        }
    }
    dag.count_saturated = crate::rashomon_dag::dag_has_saturated_count(&dag);
    Ok(LoadedDag {
        dag,
        feature_names,
        feature_types,
        fit_config,
        n_train,
    })
}

/// v4 body: dedup tables, fixed subproblem records, root.
fn read_body_v4(
    cur: &mut Cursor,
    n_subproblems: usize,
    n_thresholds: usize,
    n_group_maps: usize,
    n_root_entries: usize,
    has_group_maps: bool,
) -> Result<
    (
        Vec<FactoredSubproblem>,
        Vec<SplitSpec>,
        FSubRef,
        Vec<CostEntry>,
    ),
    String,
> {
    // DEDUP TABLES
    let mut thresholds: Vec<Arc<[f64]>> = Vec::with_capacity(n_thresholds.min(cur.remaining()));
    for _ in 0..n_thresholds {
        let len = cur.read_u32()? as usize;
        let mut v: Vec<f64> = Vec::with_capacity(len.min(cur.remaining()));
        for _ in 0..len {
            v.push(cur.read_f64()?);
        }
        thresholds.push(Arc::from(v.into_boxed_slice()));
    }

    let mut group_maps: Vec<Arc<[u16]>> = Vec::with_capacity(n_group_maps.min(cur.remaining()));
    if has_group_maps {
        for _ in 0..n_group_maps {
            let len = cur.read_u32()? as usize;
            let mut v: Vec<u16> = Vec::with_capacity(len.min(cur.remaining()));
            for _ in 0..len {
                v.push(cur.read_u16()?);
            }
            group_maps.push(Arc::from(v.into_boxed_slice()));
        }
    }

    // SUBPROBLEMS
    // Rebuild the SplitSpec pool on load. The on-disk format
    // still stores the resolved spec fields inline per split; we intern
    // them into a deduplicated pool and store a spec_id on each split.
    let mut spec_pool = crate::types::SplitSpecPool::new();
    let mut subproblems: Vec<FactoredSubproblem> =
        Vec::with_capacity(n_subproblems.min(cur.remaining()));
    for sp_idx in 0..n_subproblems {
        let sp = read_subproblem(cur, &thresholds, &group_maps, sp_idx, &mut spec_pool)?;
        subproblems.push(sp);
    }
    let spec_pool_vec = spec_pool.into_pool();

    // ROOT
    let root_ref: FSubRef = cur.read_u32()?;
    if (root_ref as usize) >= subproblems.len() {
        return Err(format!(
            "root_ref {} out of bounds (n_subproblems={})",
            root_ref,
            subproblems.len()
        ));
    }
    let mut root_cost_schedule: Vec<CostEntry> =
        Vec::with_capacity(n_root_entries.min(cur.remaining()));
    for _ in 0..n_root_entries {
        let cost = cur.read_f64()?;
        let tc = cur.read_u128()?;
        root_cost_schedule.push(CostEntry::new(cost, tc));
    }

    Ok((subproblems, spec_pool_vec, root_ref, root_cost_schedule))
}

/// Section sizes of a v5 file (from its header).
struct V5Counts {
    n_subproblems: usize,
    n_thresholds: usize,
    n_group_maps: usize,
    n_root_entries: usize,
    n_specs: usize,
    n_split_records: usize,
}

/// v5 body (layout in the module docs). Splits are expanded inline per
/// vertex from the shared record table (as v4 loads them). Without stored
/// counts every tree_count is 0 and the root schedule empty: the caller
/// recounts.
fn read_body_v5(
    cur: &mut Cursor,
    n: V5Counts,
    counts_stored: bool,
) -> Result<
    (
        Vec<FactoredSubproblem>,
        Vec<SplitSpec>,
        FSubRef,
        Vec<CostEntry>,
    ),
    String,
> {
    let mut thresholds: Vec<Arc<[f64]>> = Vec::with_capacity(n.n_thresholds.min(cur.remaining()));
    for _ in 0..n.n_thresholds {
        let len = cur.read_varint_u32()? as usize;
        let mut v: Vec<f64> = Vec::with_capacity(len.min(cur.remaining()));
        for _ in 0..len {
            v.push(cur.read_f64()?);
        }
        thresholds.push(Arc::from(v.into_boxed_slice()));
    }
    let mut group_maps: Vec<Arc<[u16]>> = Vec::with_capacity(n.n_group_maps.min(cur.remaining()));
    for _ in 0..n.n_group_maps {
        let len = cur.read_varint_u32()? as usize;
        let mut v: Vec<u16> = Vec::with_capacity(len.min(cur.remaining()));
        for _ in 0..len {
            let g = cur.read_varint_u32()?;
            v.push(u16::try_from(g).map_err(|_| format!("group_map entry {} exceeds u16", g))?);
        }
        group_maps.push(Arc::from(v.into_boxed_slice()));
    }
    let mut spec_pool: Vec<SplitSpec> = Vec::with_capacity(n.n_specs.min(cur.remaining()));
    for i in 0..n.n_specs {
        let feature_index = u16::try_from(cur.read_varint_u32()?)
            .map_err(|_| format!("spec {}: feature index exceeds u16", i))?;
        let arity = cur.read_u8()?;
        let tid = cur.read_varint_u32()? as usize;
        let gid = cur.read_varint_u32()? as usize;
        let thresholds_arc = thresholds
            .get(tid)
            .cloned()
            .ok_or_else(|| format!("spec {}: threshold id {} out of bounds", i, tid))?;
        let group_map = if gid == 0 {
            None
        } else {
            Some(
                group_maps
                    .get(gid - 1)
                    .cloned()
                    .ok_or_else(|| format!("spec {}: group_map id {} out of bounds", i, gid - 1))?,
            )
        };
        spec_pool.push(SplitSpec {
            feature_index,
            arity,
            thresholds: thresholds_arc,
            group_map,
        });
    }
    let n_leaf_values = cur.read_varint_u32()? as usize;
    let mut leaf_values: Vec<(f64, f64)> = Vec::with_capacity(n_leaf_values.min(cur.remaining()));
    for _ in 0..n_leaf_values {
        let c = cur.read_f64()?;
        let l = cur.read_f64()?;
        leaf_values.push((c, l));
    }
    let mut records: Vec<FactoredSplit> =
        Vec::with_capacity(n.n_split_records.min(cur.remaining()));
    for i in 0..n.n_split_records {
        let spec_id = cur.read_varint_u32()?;
        let arity = spec_pool
            .get(spec_id as usize)
            .ok_or_else(|| format!("split record {}: spec id {} out of bounds", i, spec_id))?
            .arity as usize;
        let mut children: crate::types::InlineChildren = smallvec::SmallVec::with_capacity(arity);
        for _ in 0..arity {
            let c = cur.read_varint_u32()?;
            if c as usize >= n.n_subproblems {
                return Err(format!("split record {}: child {} out of bounds", i, c));
            }
            children.push(c);
        }
        let tree_count = if counts_stored { cur.read_varint()? } else { 0 };
        records.push(FactoredSplit::new(spec_id, children, tree_count));
    }
    let mut subproblems: Vec<FactoredSubproblem> =
        Vec::with_capacity(n.n_subproblems.min(cur.remaining()));
    for i in 0..n.n_subproblems {
        let vflags = cur.read_u8()?;
        let leaf = if vflags & 1 != 0 {
            let prediction = cur.read_u8()?;
            let lid = cur.read_varint_u32()? as usize;
            let n_samples = cur.read_varint_u64()? as usize;
            let &(cost, loss) = leaf_values
                .get(lid)
                .ok_or_else(|| format!("vertex {}: leaf value id {} out of bounds", i, lid))?;
            Some((
                cost,
                RLeafInfo {
                    prediction,
                    loss,
                    n_samples,
                },
            ))
        } else {
            None
        };
        let n_splits = cur.read_varint_u32()? as usize;
        let mut splits: Vec<FactoredSplit> = Vec::with_capacity(n_splits.min(cur.remaining()));
        for _ in 0..n_splits {
            let id = cur.read_varint_u32()? as usize;
            splits.push(
                records
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("vertex {}: split record {} out of bounds", i, id))?,
            );
        }
        let tree_count = if counts_stored { cur.read_varint()? } else { 0 };
        subproblems.push(FactoredSubproblem {
            leaf,
            leaf_minority: None,
            prefix_leaf: None,
            splits,
            split_ids: Vec::new(),
            tree_count,
            // min_obj: recomputed after load (recount or `recompute_min_obj_arena`).
            min_obj: f64::INFINITY,
            cost_entries: None,
        });
    }
    let root_ref = cur.read_varint_u32()?;
    if (root_ref as usize) >= subproblems.len() {
        return Err(format!(
            "root_ref {} out of bounds (n_subproblems={})",
            root_ref,
            subproblems.len()
        ));
    }
    let mut root_cost_schedule: Vec<CostEntry> = Vec::new();
    if counts_stored {
        root_cost_schedule.reserve(n.n_root_entries);
        for _ in 0..n.n_root_entries {
            let cost = cur.read_f64()?;
            let tc = cur.read_varint()?;
            root_cost_schedule.push(CostEntry::new(cost, tc));
        }
    }
    Ok((subproblems, spec_pool, root_ref, root_cost_schedule))
}

/// v6 header fields after the flags.
struct V6Header {
    n_subproblems: usize,
    n_leaf_only: usize,
    n_internal: usize,
    n_root_entries: usize,
    codec: u8,
    n_streams: u8,
    n_specs: usize,
}

/// v6 METADATA_V6 block.
struct MetadataV6 {
    leaf_reg: f64,
    leaf_denom: u64,
    n_train: u64,
    root_ref: u32,
    n_leaf_escapes: u64,
}

/// One v6 stream, decompressed. A DEFLATE stream's declared raw length is
/// refused above DEFLATE's maximum expansion of its compressed length
/// (decompression bombs), and the inflated length must match it.
fn read_stream(cur: &mut Cursor, codec: u8, name: &str) -> Result<Vec<u8>, String> {
    let raw_len = usize::try_from(cur.read_varint_u64()?)
        .map_err(|_| format!("stream {}: raw length exceeds usize", name))?;
    let comp_len = usize::try_from(cur.read_varint_u64()?)
        .map_err(|_| format!("stream {}: compressed length exceeds usize", name))?;
    let comp = cur.read_bytes(comp_len)?;
    match codec {
        CODEC_NONE => {
            if comp_len != raw_len {
                return Err(format!(
                    "stream {}: codec none but raw length {} != stored length {}",
                    name, raw_len, comp_len
                ));
            }
            Ok(comp.to_vec())
        }
        CODEC_DEFLATE => {
            use std::io::Read;
            let max_raw = comp_len.saturating_mul(1032).saturating_add(1024);
            if raw_len > max_raw {
                return Err(format!(
                    "stream {}: raw length {} out of bounds for {} compressed bytes",
                    name, raw_len, comp_len
                ));
            }
            // Declared lengths are untrusted: pre-allocate a bounded amount
            // and let `read_to_end` grow the buffer.
            let mut out =
                Vec::with_capacity(raw_len.min(comp_len.saturating_mul(8).saturating_add(4096)));
            flate2::read::DeflateDecoder::new(comp)
                .take(raw_len as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|e| format!("stream {}: inflate failed: {}", name, e))?;
            if out.len() != raw_len {
                return Err(format!(
                    "stream {}: inflated {} bytes, header says {}",
                    name,
                    out.len(),
                    raw_len
                ));
            }
            Ok(out)
        }
        c => Err(format!("unknown v6 stream codec {}", c)),
    }
}

/// Error unless a v6 stream cursor is fully consumed.
fn v6_stream_done(cur: &Cursor, name: &str) -> Result<(), String> {
    if cur.pos != cur.data.len() {
        return Err(format!(
            "stream {}: {} trailing bytes",
            name,
            cur.data.len() - cur.pos
        ));
    }
    Ok(())
}

/// Leaf-option reader over the v6 PRED / KCODE / NSAMP streams.
struct V6LeafReader<'a> {
    pred: Cursor<'a>,
    kcode: Cursor<'a>,
    nsamp: Cursor<'a>,
    escapes: Vec<(f64, f64)>,
    leaf_reg: f64,
    leaf_denom: u64,
}

impl V6LeafReader<'_> {
    fn read(&mut self) -> Result<(f64, RLeafInfo), String> {
        let prediction = self.pred.read_u8()?;
        let code = self.kcode.read_varint_u64()?;
        let (cost, loss) = if code & 1 == 0 {
            if self.leaf_denom == 0 {
                return Err("integer leaf code with leaf_denom 0".into());
            }
            let loss = ((code >> 1) as f64) / (self.leaf_denom as f64);
            (loss + self.leaf_reg, loss)
        } else {
            let j = (code >> 1) as usize;
            *self.escapes.get(j).ok_or_else(|| {
                format!(
                    "leaf escape {} out of bounds (have {})",
                    j,
                    self.escapes.len()
                )
            })?
        };
        let n_samples = usize::try_from(self.nsamp.read_varint_u64()?)
            .map_err(|_| "leaf n_samples exceeds usize".to_string())?;
        Ok((
            cost,
            RLeafInfo {
                prediction,
                loss,
                n_samples,
            },
        ))
    }
}

/// v6 body (layout in the module docs): the 12 or 13 streams after
/// METADATA_V6. Splits are expanded inline per vertex (as v4 / v5 load
/// them). Without stored counts every tree_count is 0 and the root
/// schedule empty: the caller recounts.
fn read_body_v6(
    cur: &mut Cursor,
    hdr: &V6Header,
    md6: &MetadataV6,
    has_group_maps: bool,
    counts_stored: bool,
) -> Result<
    (
        Vec<FactoredSubproblem>,
        Vec<SplitSpec>,
        FSubRef,
        Vec<CostEntry>,
    ),
    String,
> {
    let n_streams = V6_N_STREAMS + counts_stored as u8;
    if hdr.n_streams != n_streams {
        return Err(format!(
            "v6: {} streams, expected {}",
            hdr.n_streams, n_streams
        ));
    }
    if hdr.n_leaf_only.checked_add(hdr.n_internal) != Some(hdr.n_subproblems) {
        return Err(format!(
            "v6: n_leaf_only {} + n_internal {} != n_subproblems {}",
            hdr.n_leaf_only, hdr.n_internal, hdr.n_subproblems
        ));
    }
    let mut raw: Vec<Vec<u8>> = Vec::with_capacity(n_streams as usize);
    const NAMES: [&str; 13] = [
        "GRID", "GMAPS", "SPECS", "VHDR", "PRED", "KCODE", "K_ESC", "NSAMP", "SPEC_IDS", "TAGS",
        "LREF", "IREF", "COUNTS",
    ];
    for name in NAMES.iter().take(n_streams as usize) {
        raw.push(read_stream(cur, hdr.codec, name)?);
    }

    // GRID: per-feature ascending threshold values.
    let mut c = Cursor::new(&raw[0]);
    let n_grid = c.read_varint_u32()? as usize;
    if n_grid > raw[0].len() {
        return Err(format!("v6 GRID: {} features out of bounds", n_grid));
    }
    let mut grid: Vec<Vec<f64>> = Vec::with_capacity(n_grid);
    for _ in 0..n_grid {
        let len = c.read_varint_u32()? as usize;
        c.need(len.saturating_mul(8))?;
        grid.push((0..len).map(|_| c.read_f64()).collect::<Result<_, _>>()?);
    }
    v6_stream_done(&c, "GRID")?;

    // GMAPS
    let mut group_maps: Vec<Arc<[u16]>> = Vec::new();
    let mut c = Cursor::new(&raw[1]);
    if has_group_maps {
        let n_gm = c.read_varint_u32()? as usize;
        for _ in 0..n_gm {
            let len = c.read_varint_u32()? as usize;
            c.need(len)?;
            let mut v: Vec<u16> = Vec::with_capacity(len);
            for _ in 0..len {
                let g = c.read_varint_u32()?;
                v.push(u16::try_from(g).map_err(|_| format!("group_map entry {} exceeds u16", g))?);
            }
            group_maps.push(Arc::from(v.into_boxed_slice()));
        }
    }
    v6_stream_done(&c, "GMAPS")?;

    // SPECS (rank order = new spec id). Equal threshold arrays share one Arc.
    let mut c = Cursor::new(&raw[2]);
    let mut thr_arcs: HashMap<Vec<u64>, Arc<[f64]>> = HashMap::new();
    let mut spec_pool: Vec<SplitSpec> = Vec::with_capacity(hdr.n_specs.min(raw[2].len()));
    for i in 0..hdr.n_specs {
        let f = c.read_varint_u32()? as usize;
        let feature_index =
            u16::try_from(f).map_err(|_| format!("spec {}: feature index exceeds u16", i))?;
        let arity = c.read_u8()?;
        let n_thr = c.read_varint_u32()? as usize;
        let g = grid
            .get(f)
            .ok_or_else(|| format!("spec {}: feature {} out of bounds of the grid", i, f))?;
        if n_thr > g.len() {
            return Err(format!(
                "spec {}: {} thresholds out of bounds (grid {})",
                i,
                n_thr,
                g.len()
            ));
        }
        let mut thr: Vec<f64> = Vec::with_capacity(n_thr);
        let mut idx: u64 = 0;
        for j in 0..n_thr {
            let gap = c.read_varint_u64()?;
            idx = if j == 0 {
                gap
            } else {
                idx.saturating_add(gap).saturating_add(1)
            };
            thr.push(*g.get(idx as usize).ok_or_else(|| {
                format!(
                    "spec {}: grid index {} out of bounds (grid {})",
                    i,
                    idx,
                    g.len()
                )
            })?);
        }
        let key: Vec<u64> = thr.iter().map(|t| t.to_bits()).collect();
        let thresholds = thr_arcs
            .entry(key)
            .or_insert_with(|| Arc::from(thr.into_boxed_slice()))
            .clone();
        let gid = c.read_varint_u32()? as usize;
        let group_map = if gid == 0 {
            None
        } else {
            Some(
                group_maps
                    .get(gid - 1)
                    .cloned()
                    .ok_or_else(|| format!("spec {}: group_map id {} out of bounds", i, gid - 1))?,
            )
        };
        spec_pool.push(SplitSpec {
            feature_index,
            arity,
            thresholds,
            group_map,
        });
    }
    v6_stream_done(&c, "SPECS")?;

    // K_ESC
    let n_esc = usize::try_from(md6.n_leaf_escapes).unwrap_or(usize::MAX);
    if n_esc.checked_mul(16) != Some(raw[6].len()) {
        return Err(format!(
            "v6 K_ESC: {} bytes for {} escapes",
            raw[6].len(),
            md6.n_leaf_escapes
        ));
    }
    let mut c = Cursor::new(&raw[6]);
    let mut escapes: Vec<(f64, f64)> = Vec::with_capacity(n_esc);
    for _ in 0..n_esc {
        escapes.push((c.read_f64()?, c.read_f64()?));
    }

    if hdr.n_leaf_only > raw[4].len() || hdr.n_internal > raw[3].len() {
        return Err(format!(
            "v6: {} leaf-only / {} internal vertices out of bounds of the streams",
            hdr.n_leaf_only, hdr.n_internal
        ));
    }
    let mut leaves = V6LeafReader {
        pred: Cursor::new(&raw[4]),
        kcode: Cursor::new(&raw[5]),
        nsamp: Cursor::new(&raw[7]),
        escapes,
        leaf_reg: md6.leaf_reg,
        leaf_denom: md6.leaf_denom,
    };
    let empty_sub = |leaf, splits| FactoredSubproblem {
        leaf,
        leaf_minority: None,
        prefix_leaf: None,
        splits,
        split_ids: Vec::new(),
        tree_count: 0,
        // min_obj: recomputed after load (recount or `recompute_min_obj_arena`).
        min_obj: f64::INFINITY,
        cost_entries: None,
    };
    // Capacity hint only, capped by the decoded bytes (a header can claim any count).
    let decoded_bytes: usize = raw.iter().map(|r| r.len()).sum();
    let mut subproblems: Vec<FactoredSubproblem> =
        Vec::with_capacity(hdr.n_subproblems.min(decoded_bytes));
    for _ in 0..hdr.n_leaf_only {
        subproblems.push(empty_sub(Some(leaves.read()?), Vec::new()));
    }

    // Internal vertices.
    let mut vhdr = Cursor::new(&raw[3]);
    let mut spec_ids = Cursor::new(&raw[8]);
    let tags: &[u8] = &raw[9];
    let mut lref = Cursor::new(&raw[10]);
    let mut iref = Cursor::new(&raw[11]);
    let mut n_tag_bits: usize = 0;
    for pos in 0..hdr.n_internal {
        let h = vhdr.read_varint_u64()?;
        let n_splits = (h >> 1) as usize;
        if n_splits > raw[8].len() - spec_ids.pos {
            return Err(format!(
                "v6: internal vertex {}: {} splits out of bounds",
                pos, n_splits
            ));
        }
        let leaf = if h & 1 != 0 {
            Some(leaves.read()?)
        } else {
            None
        };
        let mut splits: Vec<FactoredSplit> = Vec::with_capacity(n_splits);
        for _ in 0..n_splits {
            let rank = spec_ids.read_varint_u32()?;
            let arity = spec_pool
                .get(rank as usize)
                .ok_or_else(|| format!("v6: spec rank {} out of bounds", rank))?
                .arity as usize;
            let mut children: crate::types::InlineChildren =
                smallvec::SmallVec::with_capacity(arity);
            for _ in 0..arity {
                let byte = *tags
                    .get(n_tag_bits / 8)
                    .ok_or_else(|| "v6 TAGS: stream exhausted".to_string())?;
                let is_leaf = (byte >> (n_tag_bits % 8)) & 1 != 0;
                n_tag_bits += 1;
                if is_leaf {
                    let r = lref.read_varint_u32()?;
                    if r as usize >= hdr.n_leaf_only {
                        return Err(format!("v6: leaf ref {} out of bounds", r));
                    }
                    children.push(r);
                } else {
                    let delta = iref.read_varint_u64()?;
                    if delta == 0 || delta > pos as u64 {
                        return Err(format!(
                            "v6: internal ref delta {} out of bounds at position {}",
                            delta, pos
                        ));
                    }
                    children.push((hdr.n_leaf_only + pos - delta as usize) as FSubRef);
                }
            }
            splits.push(FactoredSplit::new(rank, children, 0));
        }
        subproblems.push(empty_sub(leaf, splits));
    }
    if tags.len() != n_tag_bits.div_ceil(8) {
        return Err(format!(
            "v6 TAGS: {} bytes for {} children",
            tags.len(),
            n_tag_bits
        ));
    }
    for (cur_s, name) in [
        (&vhdr, "VHDR"),
        (&leaves.pred, "PRED"),
        (&leaves.kcode, "KCODE"),
        (&leaves.nsamp, "NSAMP"),
        (&spec_ids, "SPEC_IDS"),
        (&lref, "LREF"),
        (&iref, "IREF"),
    ] {
        v6_stream_done(cur_s, name)?;
    }

    let root_ref = md6.root_ref;
    if (root_ref as usize) >= subproblems.len() {
        return Err(format!(
            "root_ref {} out of bounds (n_subproblems={})",
            root_ref,
            subproblems.len()
        ));
    }

    // COUNTS: per split (split order), per vertex, root schedule.
    let mut root_cost_schedule: Vec<CostEntry> = Vec::new();
    if counts_stored {
        let mut c = Cursor::new(&raw[12]);
        for sp in subproblems.iter_mut() {
            for split in sp.splits.iter_mut() {
                split.set_tree_count(c.read_varint()?);
            }
        }
        for sp in subproblems.iter_mut() {
            sp.tree_count = c.read_varint()?;
        }
        root_cost_schedule.reserve(hdr.n_root_entries.min(raw[12].len()));
        for _ in 0..hdr.n_root_entries {
            let cost = c.read_f64()?;
            let tc = c.read_varint()?;
            root_cost_schedule.push(CostEntry::new(cost, tc));
        }
        v6_stream_done(&c, "COUNTS")?;
    }
    Ok((subproblems, spec_pool, root_ref, root_cost_schedule))
}

/// Post-load recompute of `min_obj` for every subproblem in
/// the arena: min_obj = min(leaf cost, min over splits of Σ child
/// min_obj), children before parents. The order is a post-order DFS over
/// the arena, so it does not rely on the arena being children-first (a
/// full build reserves a parent's slot before its children; a Step-B graft
/// appends nested arenas after the placeholders that use them).
fn recompute_min_obj_arena(subproblems: &mut [FactoredSubproblem]) {
    let n = subproblems.len();
    let mut state = vec![0u8; n];
    let mut post: Vec<usize> = Vec::with_capacity(n);
    for start in 0..n {
        if state[start] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, bool)> = vec![(start, false)];
        while let Some((i, children_done)) = stack.pop() {
            if children_done {
                post.push(i);
                continue;
            }
            if state[i] != 0 {
                continue;
            }
            state[i] = 1;
            stack.push((i, true));
            for split in &subproblems[i].splits {
                for &c in split.children.iter() {
                    let c = c as usize;
                    if c < n && state[c] == 0 {
                        stack.push((c, false));
                    }
                }
            }
        }
    }
    for i in post {
        let mut best = f64::INFINITY;
        if let Some((leaf_cost, _)) = subproblems[i].leaf {
            if leaf_cost < best {
                best = leaf_cost;
            }
        }
        // Snapshot child min_obj values into a local Vec to avoid
        // borrow-overlap with subproblems[i].splits.
        let split_lbs: Vec<f64> = subproblems[i]
            .splits
            .iter()
            .map(|split| {
                split
                    .children
                    .iter()
                    .map(|&cref| {
                        subproblems
                            .get(cref as usize)
                            .map(|c| c.min_obj)
                            .unwrap_or(f64::INFINITY)
                    })
                    .fold(0.0_f64, |acc, v| acc + v)
            })
            .collect();
        for s in split_lbs {
            if s < best {
                best = s;
            }
        }
        subproblems[i].min_obj = best;
    }
}

fn read_subproblem(
    cur: &mut Cursor,
    thresholds: &[Arc<[f64]>],
    group_maps: &[Arc<[u16]>],
    sp_idx: usize,
    spec_pool: &mut crate::types::SplitSpecPool,
) -> Result<FactoredSubproblem, String> {
    let flags = cur.read_u8()?;
    let prediction = cur.read_u8()?;
    let n_splits = cur.read_u16()? as usize;
    let leaf_cost = cur.read_f64()?;
    let leaf_loss = cur.read_f64()?;
    let leaf_n_samples = cur.read_u64()? as usize;
    let tree_count = cur.read_u128()?;

    let leaf = if (flags & 1) != 0 {
        Some((
            leaf_cost,
            RLeafInfo {
                prediction,
                loss: leaf_loss,
                n_samples: leaf_n_samples,
            },
        ))
    } else {
        None
    };

    let mut splits: Vec<FactoredSplit> = Vec::with_capacity(n_splits);
    for _ in 0..n_splits {
        splits.push(read_split(cur, thresholds, group_maps, sp_idx, spec_pool)?);
    }

    Ok(FactoredSubproblem {
        leaf,
        // The on-disk format has no minority leaf options; loaded
        // DAGs never carry one (the serializer expands paired leaves, if
        // any, before write — see write path). Always None on load.
        leaf_minority: None,
        prefix_leaf: None,
        splits,
        split_ids: Vec::new(),
        tree_count,
        // min_obj is recomputed bottom-up post-load via
        // `recompute_min_obj_arena`; placeholder INFINITY here.
        min_obj: f64::INFINITY,
        // Fix B: cost_entries is NOT persisted in the on-disk format (it
        // would inflate file size and is derivable lazily). Subproblems
        // loaded from disk will go through the legacy materialize_schedule
        // fallback (convolution) the first time their schedule is needed.
        cost_entries: None,
    })
}

fn read_split(
    cur: &mut Cursor,
    thresholds: &[Arc<[f64]>],
    group_maps: &[Arc<[u16]>],
    sp_idx: usize,
    spec_pool: &mut crate::types::SplitSpecPool,
) -> Result<FactoredSplit, String> {
    let feature_index = cur.read_u16()?;
    let arity = cur.read_u8()?;
    let _pad = cur.read_u8()?;
    let threshold_id = cur.read_u32()? as usize;
    let group_map_id = cur.read_u32()?;
    // On-disk count is a u128, the in-memory logical width.
    let split_tc = cur.read_u128()?;

    let thresholds_arc = thresholds.get(threshold_id).cloned().ok_or_else(|| {
        format!(
            "subproblem {}: threshold_id {} out of bounds (have {})",
            sp_idx,
            threshold_id,
            thresholds.len()
        )
    })?;
    let group_map_arc = if group_map_id == NO_GROUP_MAP {
        None
    } else {
        Some(
            group_maps
                .get(group_map_id as usize)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "subproblem {}: group_map_id {} out of bounds (have {})",
                        sp_idx,
                        group_map_id,
                        group_maps.len()
                    )
                })?,
        )
    };

    let spec = SplitSpec {
        feature_index,
        arity,
        thresholds: thresholds_arc,
        group_map: group_map_arc,
    };
    // Intern into the load-time pool → spec_id.
    let spec_id = spec_pool.intern(&spec);

    // v4+: per-child schedules are not stored; just read FSubRef.
    // Children are an inline SmallVec; push directly (k≤4 inline).
    let mut children: crate::types::InlineChildren =
        smallvec::SmallVec::with_capacity(arity as usize);
    for _ in 0..arity {
        let fsub_ref = cur.read_u32()?;
        children.push(fsub_ref);
    }

    Ok(FactoredSplit::new(spec_id, children, split_tc))
}

fn feature_type_to_u8(ft: FeatureType) -> u8 {
    match ft {
        FeatureType::Num => 0,
        FeatureType::Cat => 1,
    }
}

fn u8_to_feature_type(b: u8) -> Result<FeatureType, String> {
    match b {
        0 => Ok(FeatureType::Num),
        1 => Ok(FeatureType::Cat),
        other => Err(format!("unknown FeatureType byte: {}", other)),
    }
}

// Tests: tests/factored_dag_io_roundtrip.rs.
