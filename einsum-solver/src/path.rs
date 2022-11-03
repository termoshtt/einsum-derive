//! Construct and execute contraction path

use crate::subscripts::*;

#[cfg_attr(doc, katexit::katexit)]
/// Contraction path
///
/// Einstein summation rule
/// ------------------------
/// The Einstein summation rule in theoretical physics and related field
/// eliminates the summation symbol $\sum$ from tensor terms,
/// e.g. $\sum_{i \in I} x_i y_i$ is abbreviated into $x_i y_i$.
/// This is based on the fact that we can exchange the summation order
/// if their ranges are finite:
/// $$
/// \sum_{j \in J} \sum_{k \in K} a_{ij} b_{jk} c_{kl}
/// = \sum_{k \in K} \sum_{j \in J} a_{ij} b_{jk} c_{kl}
/// $$
/// and we can recover their range from the "type" of tensors
/// if we are ruled to sum only along all indices.
///
/// Memorize partial sum
/// ---------------------
/// Partial summation and its memorization reduces number of floating point operations.
/// For simplicity, both addition `+` and multiplication `*` are counted as 1 operation,
/// and do not consider fused multiplication-addition (FMA).
/// In the above example, there are $\\#K \times \\#J$ addition
/// and $2 \times \\#K \times \\#J$ multiplications,
/// where $\\#$ denotes the number of elements in the index sets.
///
/// When we sum up partially along `j`:
/// $$
/// \sum_{k \in K} c_{kl} \left( \sum_{j \in J} a_{ij} b_{jk} \right),
/// $$
/// and memorize its result as $d_{ik}$:
/// $$
/// \sum_{k \in K} c_{kl} d_{ik},
/// \text{where} \space d_{ik} = \sum_{j \in J} a_{ij} b_{jk},
/// $$
/// there are only $2\\#K + 2\\#J$ operations with $\\#I \times \\#K$
/// memorization storage.
///
/// Summation order
/// ----------------
/// This crate assumes that all indices are summed with memorization,
/// and this struct represents the order of summation.
/// For fixed tensor terms, the order of summation is represented
/// by a list of indices to be summed up; `['j', 'k']` means
/// first sums along `j` and then sums along `k` in above example:
/// $$
/// \sum_{k \in K} c_{kl} M_{ik}^{(j)},
/// \text{where} \space M_{ik}^{(j)} = \sum_{j \in J} a_{ij} b_{jk},
/// $$
/// and `['k', 'j']` means:
/// $$
/// \sum_{j \in J} a_{ij} M_{jl}^{(k)},
/// \text{where} \space M_{jl}^{(k)} = \sum_{k \in K} b_{jk} c_{kl}
/// $$
/// We denote the memorized partial sum tensor as $M_{ik}^{(j)}$.
///
/// We can easily find that $c_{kl}$ or $a_{ij}$ can be put out of the inner summation
/// since we know matrix-multiplication is associative $ABC = (AB)C = A(BC)$.
/// For more complicated case, e.g. computing `j` contraction of `ijk,klj,km->ilm`,
/// we will check the input indices:
///
/// - `ijk` contains `j`
/// - `klj` contains `j`
/// - `km` does not contain `j`
///
/// and then compute the indices of intermediate tensor by
/// $$
/// \left(
///   \\{ i, j, k \\} \cup \\{ k, l, j\\}
/// \right)
/// \setminus j
/// = \\{ i, k, l\\},
/// $$
/// i.e. memorize $M_{ikl}^{(j)}$ instead of $M_{iklm}^{(j)}$.
///
#[derive(Debug, PartialEq, Eq)]
pub struct Path {
    pub subscripts: Subscripts,
    pub path: Vec<char>,
}

impl Path {
    /// Manually set contraction order
    pub fn manual(subscripts: Subscripts, path: Vec<char>) -> Self {
        Path { subscripts, path }
    }

    /// Alphabetical order
    ///
    /// ```
    /// use std::str::FromStr;
    /// use einsum_solver::{path::Path, subscripts::Subscripts};
    ///
    /// let subscripts = Subscripts::from_str("ij,ji->").unwrap();
    /// let path = Path::alphabetical(subscripts);
    /// assert_eq!(path.path, &['i', 'j']);
    /// ```
    pub fn alphabetical(subscripts: Subscripts) -> Self {
        let path = subscripts.contraction_indices().into_iter().collect();
        Path { subscripts, path }
    }
}
