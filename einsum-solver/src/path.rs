//! Construct and execute contraction path

use crate::subscripts::*;

/// Contraction path
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
    /// use einsum_solver::{path::Path, subscripts::{Subscripts, Namespace}};
    ///
    /// let mut names = Namespace::init();
    /// let subscripts = Subscripts::from_raw_indices(&mut names, "ij,ji->").unwrap();
    /// let path = Path::alphabetical(subscripts);
    /// assert_eq!(path.path, &['i', 'j']);
    /// ```
    pub fn alphabetical(subscripts: Subscripts) -> Self {
        let path = subscripts.contraction_indices().into_iter().collect();
        Path { subscripts, path }
    }
}
