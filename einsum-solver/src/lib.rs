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
//! For example, `inner` is represented by `i,i->`, `matmul` is represented by `ij,jk->ik`,
//! `matmul3` is represented by `ij,jk,kl->il`, and so on
//! where `,` is the separator of each indices
//! and each index must be represented by a single char like `i` or `j`.
//! `->` separates the indices of input tensors and indices of output tensor.
//! If no indices are placed like `i,i->`, it means the tensor is 0-rank, i.e. a scalar value.
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
//! This is written as `ij,ji->` in einsum subscript form.
//! We cannot factor out both $a_{ij}$ and $b_{ji}$ out of summation
//! since they contain both indices.
//! Whether this factorization is possible or not can be determined only
//! from einsum subscript form, and we call a subscript is "reducible"
//! if factorization is possible, and "irreducible" if not possible,
//! i.e. `ij,jk,kl->il` is reducible and `ij,ji->` is irreducible.
//!
//! ### Subscript representation
//!
//! To discuss subscript factorization, we have to track which tensors are
//! used as each input.
//! In above `matmul3` example, `ij,jk,kl->il` is factorized into sub-subscripts
//! `ij,jk->ik` and `ik,kl->il` where `ik` in the second subscript uses
//! the output of first subscript. The information of the name of tensors
//! has been dropped from sub-subscripts,
//! and we have to create a mechanism for managing it.
//!
//! We introduce a subscript representation of `matmul3` case with tensor names:
//!
//! ```text
//! ij,jk,kl->il | a b c -> out
//! ```
//!
//! In this form, the factorization can be described:
//!
//! ```text
//! ij,jk->ik | a b -> d
//! ik,kl->il | d c -> out
//! ```
//!
//! To clarify the tensor is given from user or created while factorization,
//! we use `arg{N}` and `out{N}` identifiers:
//!
//! ```text
//! ij,jk->ik | arg0 arg1 -> out0
//! ik,kl->il | out0 arg2 -> out1
//! ```
//!
//! ### Summation order
//!
//! This factorization is not unique.
//! Apparently, there are two ways for `matmul3` case as corresponding to $(AB)C$:
//!
//! ```text
//! ij,jk->ik | arg0 arg1 -> out0
//! ik,kl->il | out0 arg2 -> out1
//! ```
//!
//! and to $A(BC)$:
//!
//! ```text
//! jk,kl->jl | arg1 arg2 -> out0
//! jl,ij->il | out0 arg0 -> out1
//! ```
//!
//! These are different procedure i.e. number of floating operations
//! and required intermediate memories are different,
//! but return same output tensor
//! (we ignore non-associativity of floating numbers on this document).
//! This becomes complicated combinational optimization problem
//! if there are many contraction indicies,
//! and the objective of this crate is to (heuristically) solve this problem.
//!

pub mod parser;
pub mod path;
pub mod subscripts;
