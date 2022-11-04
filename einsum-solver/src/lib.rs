#![cfg_attr(
    doc,
    feature(prelude_import, custom_inner_attributes, proc_macro_hygiene)
)]
#![cfg_attr(doc, katexit::katexit)]
//! Helper crate for einsum algorithm
//!
//! Einstein summation rule
//! ------------------------
//! The Einstein summation rule in theoretical physics and related field
//! including machine learning is a rule for abbreviating tensor operations.
//! For example, one of most basic tensor operation is inner product of
//! two vectors in $n$-dimensional Euclidean space $x, y \in \mathbb{R}^n$:
//! $$
//! (x, y) = \sum_{i \in I} x_i y_i
//! $$
//! where $I$ denotes a set of indices, i.e. $I = \\{0, 1, \ldots, n-1 \\}$.
//! Another example is matrix multiplications.
//! A multiplication of three square matrices $A, B, C \in M_n(\mathbb{R})$
//! can be written as its element:
//! $$
//! ABC_{il} = \sum_{j \in J} \sum_{k \in K} a_{ij} b_{jk} c_{kl}
//! $$
//!
//! Many such tensor operations appear in various field,
//! and we usually define many functions corresponding to each operations.
//! For inner product of vectors, we may defines a function like
//! ```ignore
//! fn inner(a: Array1D<R>, b: Array1D<R>) -> R;
//! ```
//! for matrix multiplication:
//! ```ignore
//! fn matmul(a: Array2D<R>, b: Array2D<R>) -> Array2D<R>;
//! ```
//! or taking three matrices:
//! ```ignore
//! fn matmul3(a: Array2D<R>, b: Array2D<R>, c: Array2D<R>) -> Array2D<R>;
//! ```
//! and so on.
//!
//! These definitions are very similar, and actually,
//! they can be represented in a single manner.
//!
//! einsum algorithm
//! -----------------
//! We discuss an overview of einsum algorithm for understanding the structure of this crate.
//!
//! ### Memorize partial sum
//! Partial summation and its memorization reduces number of floating point operations.
//! For simplicity, both addition `+` and multiplication `*` are counted as 1 operation,
//! and do not consider fused multiplication-addition (FMA).
//! In the above example, there are $\\#K \times \\#J$ addition
//! and $2 \times \\#K \times \\#J$ multiplications,
//! where $\\#$ denotes the number of elements in the index sets.
//!
//! When we sum up partially along `j`:
//! $$
//! \sum_{k \in K} c_{kl} \left( \sum_{j \in J} a_{ij} b_{jk} \right),
//! $$
//! and memorize its result as $d_{ik}$:
//! $$
//! \sum_{k \in K} c_{kl} d_{ik},
//! \text{where} \space d_{ik} = \sum_{j \in J} a_{ij} b_{jk},
//! $$
//! there are only $2\\#K + 2\\#J$ operations with $\\#I \times \\#K$
//! memorization storage.
//!
//! ### Summation order
//! This crate assumes that all indices are summed with memorization,
//! and this struct represents the order of summation.
//! For fixed tensor terms, the order of summation is represented
//! by a list of indices to be summed up; `['j', 'k']` means
//! first sums along `j` and then sums along `k` in above example:
//! $$
//! \sum_{k \in K} c_{kl} M_{ik}^{(j)},
//! \text{where} \space M_{ik}^{(j)} = \sum_{j \in J} a_{ij} b_{jk},
//! $$
//! and `['k', 'j']` means:
//! $$
//! \sum_{j \in J} a_{ij} M_{jl}^{(k)},
//! \text{where} \space M_{jl}^{(k)} = \sum_{k \in K} b_{jk} c_{kl}
//! $$
//! We denote the memorized partial sum tensor as $M_{ik}^{(j)}$.
//!
//! We can easily find that $c_{kl}$ or $a_{ij}$ can be put out of the inner summation
//! since we know matrix-multiplication is associative $ABC = (AB)C = A(BC)$.
//! For more complicated case, e.g. computing `j` contraction of `ijk,klj,km->ilm`,
//! we will check the input indices:
//!
//! - `ijk` contains `j`
//! - `klj` contains `j`
//! - `km` does not contain `j`
//!
//! and then compute the indices of intermediate tensor by
//! $$
//! \left(
//!   \\{ i, j, k \\} \cup \\{ k, l, j\\}
//! \right)
//! \setminus j
//! = \\{ i, k, l\\},
//! $$
//! i.e. memorize $M_{ikl}^{(j)}$ instead of $M_{iklm}^{(j)}$.
//!

pub mod error;
pub mod parser;
pub mod path;
pub mod subscripts;
