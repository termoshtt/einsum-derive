#![cfg_attr(
    doc,
    feature(prelude_import, custom_inner_attributes, proc_macro_hygiene)
)]
#![cfg_attr(doc, katexit::katexit)]
//! Helper crate for einsum algorithm
//!
//! Introduction to einsum
//! -----------------------
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
//! These computations multiplicate the element of each tensor with some indices,
//! and sum up them along some indices.
//! So we have to determine
//!
//! - what indices to be used for each tensors in multiplications
//! - what indices to be summed up
//!
//! This can be done by ordering indices for input tensors
//! with a Einstein summation rule, i.e. sum up indices which appears more than once.
//! For example, `inner` is represented by `i,i`, `matmul` is represented by `ij,jk`,
//! `matmul3` is represented by `ij,jk,kl`, and so on
//! where `,` is the separator of each indices
//! and each index must be represented by a single char like `i` or `j`.
//! "einsum" is an algorithm or runtime to be expand such string
//! into actual tensor operations.
//!
//! einsum algorithm
//! -----------------
//! We discuss an overview of einsum algorithm for understanding the structure of this crate.
//!
//! ### Factorize and Memorize partial summation
//! Partial summation and its memorization reduces number of floating point operations.
//! For simplicity, both addition `+` and multiplication `*` are counted as 1 operation,
//! and do not consider fused multiplication-addition (FMA).
//! In the above `matmul3` example, there are $\\#K \times \\#J$ addition
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
//! When is this factorization possible? We know that above `matmul3` example
//! is also written as associative matrix product $ABC = A(BC) = (AB)C$,
//! and partial summation along $j$ is corresponding to store $D = AB$.
//! This is not always possible. Let us consider a trace of two matrix product
//! $$
//! \text{Tr} (AB) = \sum_{i \in I} \sum_{j \in J} a_{ij} b_{ji}
//! $$
//! This is written as `ij,ji` in einsum subscript form.
//! We cannot factor out both $a_{ij}$ and $b_{ji}$ out of summation
//! since they contain both indices.
//! Whether this factorization is possible or not can be determined only
//! from einsum subscript form, and we call a subscript is "reducible"
//! if factorization is possible, and "irreducible" if not possible,
//! i.e. `ij,jk,kl` is reducible and `ij,ji` is irreducible.
//!
//! ### Subscript representation
//!
//! We introduce a subscript representation for this decomposition process.
//! First, the user input for `matmul3` case is written as following:
//!
//! ```text
//! 0 | ij,jk,kl->il | a b c -> out
//! ```
//!
//! This is then decomposed into two primitive processes:
//!
//! ```text
//! 0 | ij,jk->ik | a b -> d
//! 1 | ik,kl->il | d c -> out
//! ```
//!
//! Each line is called "step" and represents a computation process.
//!
//! - Left columns represents "step id"
//! - Center column represents what computation is executed
//! - Right column represents which tensors are used and created on the step
//!
//! If such decomposition are possible for a subscripts, we call it reducible,
//! and call it irreducible if not.
//! When there are more than two indices to be contracted,
//! the subscripts is reducible because it is always possible to sum up partially
//! along one of indices.
//! Inductively, if there are $n$ contraction indices, it is decomposed into
//! $n$ irreducible steps.
//!
//! ### Summation order
//!
//! This decomposition is not unique.
//! Apparently, there are two ways for `matmul3` case as corresponding to $(AB)C$:
//!
//! ```text
//! 0 | ij,jk->ik | a b -> d
//! 1 | ik,kl->il | d c -> out
//! ```
//!
//! and to $A(BC)$:
//!
//! ```text
//! 0 | jk,kl->jl | b c -> d
//! 1 | jl,ij->il | d a -> out
//! ```
//!
//! ### Intermediate tensors
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
