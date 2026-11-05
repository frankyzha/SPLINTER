use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::types::PyList;

#[cfg(feature = "python")]
use crate::types::FeatureType;

/// Tree node representation for extraction and conversion to Python.
#[derive(Clone)]
pub enum TreeNode {
    Leaf {
        prediction: usize,
        n_samples: usize,
        loss: f64,
    },
    Internal {
        feature_index: usize,
        thresholds: Arc<[f64]>,
        children: Vec<TreeNode>,
        /// Shape mode: maps bin index -> child index.
        group_map: Option<Vec<usize>>,
    },
}

/// Convert a TreeNode to a Python dict, matching the MWNode/MWLeaf structure.
///
/// Leaf dict: {"type": "leaf", "prediction": int, "n_samples": int, "loss": float}
/// Node dict: {"type": "node", "feature_index": int, "feature_name": str,
///              "thresholds": list[float], "children": list[dict]}
#[cfg(feature = "python")]
pub fn tree_to_pydict<'py>(
    py: Python<'py>,
    node: &TreeNode,
    feature_names: &[String],
    feature_types: &[FeatureType],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    match node {
        TreeNode::Leaf {
            prediction,
            n_samples,
            loss,
        } => {
            dict.set_item("type", "leaf")?;
            dict.set_item("prediction", *prediction)?;
            dict.set_item("n_samples", *n_samples)?;
            dict.set_item("loss", *loss)?;
        }
        TreeNode::Internal {
            feature_index,
            thresholds,
            children,
            group_map,
        } => {
            dict.set_item("type", "node")?;
            dict.set_item("feature_index", *feature_index)?;
            let fname = feature_names
                .get(*feature_index)
                .map(|s| s.as_str())
                .unwrap_or("?");
            dict.set_item("feature_name", fname)?;
            dict.set_item("thresholds", thresholds.to_vec())?;

            if feature_types.get(*feature_index) == Some(&FeatureType::Cat) {
                dict.set_item("feature_type", "cat")?;
            }

            if let Some(gmap) = group_map {
                dict.set_item("group_map", gmap.clone())?;
            }

            let child_list = PyList::empty_bound(py);
            for child in children {
                let child_dict = tree_to_pydict(py, child, feature_names, feature_types)?;
                child_list.append(child_dict)?;
            }
            dict.set_item("children", child_list)?;
        }
    }
    Ok(dict)
}
