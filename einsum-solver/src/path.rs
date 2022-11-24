//! Construct and execute contraction path

use crate::{namespace::Namespace, subscripts::*};
use anyhow::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path(Vec<Subscripts>);

impl std::ops::Deref for Path {
    type Target = [Subscripts];
    fn deref(&self) -> &[Subscripts] {
        &self.0
    }
}

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
    let mut subpaths = (0..2_usize.pow(n as u32))
        .filter_map(|mut m| {
            // create combinations specifying which tensors are used
            let mut pos = BTreeSet::new();
            for i in 0..n {
                if m % 2 == 1 {
                    pos.insert(*subscripts.inputs[i].position());
                }
                m /= 2;
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
    subpaths.push(Path(vec![subscripts]));
    Ok(subpaths
        .into_iter()
        .min_by_key(|path| (path.compute_order(), path.memory_order()))
        .expect("subpath never be empty"))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn brute_force_ij_jk() -> Result<()> {
        let mut names = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut names, "ij,jk->ik")?;
        let path = brute_force(&mut names, subscripts)?;
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].to_string(), "ij,jk->ik | arg0,arg1->out0");
        Ok(())
    }

    #[test]
    fn brute_force_ij_jk_kl_l() -> Result<()> {
        let mut names = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut names, "ij,jk,kl,l->i")?;
        let path = brute_force(&mut names, subscripts)?;
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].to_string(), "kl,l->k | arg2,arg3->out1");
        assert_eq!(path[1].to_string(), "k,jk->j | out1,arg1->out2");
        assert_eq!(path[2].to_string(), "j,ij->i | out2,arg0->out0");
        Ok(())
    }

    #[test]
    fn brute_force_i_i_i() -> Result<()> {
        let mut names = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut names, "i,i,i->")?;
        let path = brute_force(&mut names, subscripts)?;
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].to_string(), "i,i,i-> | arg0,arg1,arg2->out0");
        Ok(())
    }
}
