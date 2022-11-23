//! Construct and execute contraction path

use crate::{namespace::Namespace, subscripts::*};
use anyhow::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path(Vec<Subscripts>);

impl Path {
    pub fn compute_order(&self) -> usize {
        self.0
            .iter()
            .map(|ss| ss.compute_order())
            .max()
            .expect("self.0 never be empty")
    }

    pub fn memory_order(&self) -> usize {
        self.0
            .iter()
            .map(|ss| ss.memory_order())
            .max()
            .expect("self.0 never be empty")
    }
}

pub fn brute_force(names: &mut Namespace, subscripts: Subscripts) -> Result<Path> {
    if subscripts.inputs.len() <= 2 {
        // Cannot be factorized anymore
        return Ok(Path(vec![subscripts]));
    }

    let n = subscripts.inputs.len();
    let subpaths = (0..2_usize.pow(n as u32))
        .filter_map(|mut m| {
            // create combinations specifying which tensors are used
            let mut pos = BTreeSet::new();
            for i in 0..n {
                if m % 2 == 1 {
                    pos.insert(subscripts.inputs[i].position().clone());
                }
                m = m / 2;
            }
            // At least two tensors, and not be all
            if pos.len() >= 2 && pos.len() < n {
                Some(pos)
            } else {
                None
            }
        })
        .map(|pos| {
            let mut names = names.clone();
            let (inner, outer) = subscripts.factorize(&mut names, pos)?;
            let mut sub = brute_force(&mut names, outer)?;
            sub.0.insert(0, inner);
            Ok(sub)
        })
        .collect::<Result<Vec<Path>>>()?;
    Ok(subpaths
        .into_iter()
        .min_by_key(|path| path.compute_order())
        .expect("subpath never be empty"))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn path() -> Result<()> {
        let mut names = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut names, "ij,jk,kl,l->i")?;
        let path = brute_force(&mut names, subscripts)?;
        dbg!(path);
        todo!()
    }
}
