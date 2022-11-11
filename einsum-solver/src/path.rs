//! Construct and execute contraction path

use crate::subscripts::*;
use anyhow::{bail, Result};
use std::collections::BTreeSet;

/// Contraction path
#[derive(Debug, PartialEq, Eq)]
pub struct Path {
    namespace: Namespace,
    subscripts: Subscripts,
    contraction_indices_rev: Vec<char>,
}

impl Iterator for Path {
    type Item = Subscripts;
    fn next(&mut self) -> Option<Self::Item> {
        if self.contraction_indices_rev.is_empty() {
            return None;
        }
        let c = self.contraction_indices_rev.pop().unwrap(/* never empty here */);
        match self.subscripts.factorize(&mut self.namespace, c) {
            Ok(None) => None,
            Ok(Some((first, second))) => {
                self.subscripts = second;
                Some(first)
            }
            Err(_) => unreachable!(),
        }
    }
}

impl Path {
    /// Manually set contraction order
    pub fn manual(subscripts: Subscripts, contraction_indices: &[char]) -> Result<Self> {
        let mut indices_as_set: BTreeSet<char> = BTreeSet::new();
        for &c in contraction_indices {
            if !indices_as_set.insert(c) {
                bail!("Duplicated contraction indices: {}", c);
            }
        }
        if indices_as_set != subscripts.contraction_indices() {
            bail!("Manually specified indices are not consistent to subscripts");
        }
        Ok(Path {
            namespace: Namespace::init(),
            subscripts,
            contraction_indices_rev: contraction_indices.iter().rev().cloned().collect(),
        })
    }

    /// Alphabetical order
    pub fn alphabetical(subscripts: Subscripts) -> Self {
        let contraction_indices_rev = subscripts.contraction_indices().into_iter().rev().collect();
        Path {
            namespace: Namespace::init(),
            subscripts,
            contraction_indices_rev,
        }
    }
}
