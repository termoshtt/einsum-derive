//! Construct and execute contraction path

use crate::subscripts::*;

#[cfg_attr(doc, katexit::katexit)]
/// Contraction path
///
/// Summation order
/// ---------------
///
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
/// if we are ruled to sum only against all indices.
///
/// However, we have to determine the order to evaluate these terms on computer.
/// For fixed tensor terms, the order of summation is represented
/// by a list of indices to be summed up; `['j', 'k']` means
/// first sums against `j` and then sums against `k`:
/// $$
/// \sum_{k \in K} \sum_{j \in J} a_{ij} b_{jk} c_{kl}
/// $$
/// and `['k', 'j']` means:
/// $$
/// \sum_{j \in J} \sum_{k \in K} a_{ij} b_{jk} c_{kl}
/// $$
///
/// Intermediate storage
/// --------------------
/// Intermediate tensors appear when we execute summation in some order.
/// For example, while we compute above `['j', 'k']` case, there are two tensors,
/// $j$-contracted tensor:
/// $$
/// I_{ikl}^{(j)} = \sum_{j \in J} a_{ij} b_{jk} c_{kl}
/// $$
/// and $j, k$-contracted tensor:
/// $$
/// I_{il}^{(j, k)} = \sum_{k \in K} \sum_{j \in J} a_{ij} b_{jk} c_{kl}.
/// $$
/// Another $k$-contracted tensor appears while we compute `['k', 'j']` case:
/// $$
/// I_{ijl}^{(k)} = \sum_{k \in K} a_{ij} b_{jk} c_{kl}.
/// $$
/// These tensors may has larger size than input tensors.
/// Denoting the size of set and the number of elements in tensor by $\\#$,
/// i.e. $\\#a = \\#I \times \\#J$, the size of these tensors are following:
/// $$
/// \begin{align*}
///   \\# I_{ikl}^{(j)} &= \\#I \times \\#K \times \\#L, \\\\
///   \\# I_{ijl}^{(k)} &= \\#I \times \\#J \times \\#L, \\\\
///   \\# I_{il}^{(j, k)} &= \\#I \times \\#L.
/// \end{align*}
/// $$
///
/// We may consider algorithm not to allocate large memories while computing
/// the final tensors depending on the target tensors.
/// In this case, since `['j', 'k']` case corresponds to a three matrix product
/// $(AB)C$ and `['k', 'j']` corresponds to $A(BC)$,
/// we can compute the final result by only storing $AB$ or $BC$,
/// which is smaller than $I_{ikl}^{(j)}$ or $I_{ijl}^{(k)}$.
/// In terms of $I$, this corresponds to define
/// $$
/// I_{ik}^{(j)} = \sum_{j \in J} a_{ij} b_{jk}
/// $$
/// and put $I_{ikl}^{(j)} = I_{ik}^{(j)} c_{kl}$ (we do not sum up $k$ here).
/// We can put $c_{kl}$ out of $I_{ik}^{(j)}$ because $c_{kl}$ does not contain
/// the index $j$,
/// and this elimination can be determined only from the indices of tensors.
/// For more complicated case, e.g. computing `j` contraction of `ijk,klj,km->ilm`,
/// we will check the input indices:
///
/// - `ijk` contains `j`
/// - `klj` contains `j`
/// - `km` does not contain `j`
///
/// and then compute the indices of intermediate tensor
/// $$
/// \left(
///   \\{ i, j, k \\} \cup \\{ k, l, j\\}
/// \right)
/// \setminus j
/// = \\{ i, k, l\\},
/// $$
/// to store $I_{ikl}^{(j)}$ instead of $I_{iklm}^{(j)}$.
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
