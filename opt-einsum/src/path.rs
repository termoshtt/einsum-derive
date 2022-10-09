//! Construct and execute contration path

use crate::parser::*;

#[cfg_attr(doc, katexit::katexit)]
/// Contraction path
///
/// The Einstein summation rule in theoretical physics and related field
/// eliminates the summation symbol $\sum$ from ternsor terms,
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
/// However, we have to determine the order to evaluate these terms.
/// For fixed tensor terms, the order is represnted by a list of indices;
/// `['j', 'k']` means
/// $$
/// \sum_{k \in K} \sum_{j \in J} a_{ij} b_{jk} c_{kl}
/// $$
/// and `['k', 'j']` means
/// $$
/// \sum_{j \in J} \sum_{k \in K} a_{ij} b_{jk} c_{kl}
/// $$
///
#[derive(Debug, PartialEq, Eq)]
pub struct Path {
    pub path: Vec<char>,
}

impl Path {
    /// Naive order, first appears first sum
    ///
    /// ```
    /// use std::str::FromStr;
    /// use opt_einsum::{path::Path, parser::SubScripts};
    ///
    /// let subscripts = SubScripts::from_str("ij,ji->").unwrap();
    /// let path = Path::naive(&subscripts);
    /// assert_eq!(path, Path { path: vec!['i', 'j'] });
    /// ```
    pub fn naive(subscripts: &SubScripts) -> Self {
        let mut count: Vec<(char, usize)> = Vec::new();
        for input in &subscripts.inputs {
            for label in input {
                match label {
                    Label::Index(index) => {
                        let mut found = false;
                        for (c, count) in &mut count {
                            if index == c {
                                *count += 1;
                                found = true;
                            }
                        }
                        if !found {
                            count.push((*index, 1))
                        }
                    }
                    Label::Ellipsis => unimplemented!("Ellipsis (...) is not supported yet"),
                };
            }
        }
        Path {
            path: count
                .into_iter()
                .flat_map(|(key, value)| if value > 1 { Some(key) } else { None })
                .collect(),
        }
    }
}
