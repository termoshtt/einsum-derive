//! Execution path

use crate::*;
use anyhow::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    original: Subscripts,
    reduced_subscripts: Vec<Subscripts>,
}

impl std::ops::Deref for Path {
    type Target = [Subscripts];
    fn deref(&self) -> &[Subscripts] {
        &self.reduced_subscripts
    }
}

impl Path {
    pub fn output(&self) -> &Subscript {
        &self.original.output
    }

    pub fn num_args(&self) -> usize {
        self.original.inputs.len()
    }

    pub fn compute_order(&self) -> usize {
        compute_order(&self.reduced_subscripts)
    }

    pub fn memory_order(&self) -> usize {
        memory_order(&self.reduced_subscripts)
    }

    pub fn brute_force(indices: &str) -> Result<Self> {
        let mut names = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut names, indices)?;
        Ok(Path {
            original: subscripts.clone(),
            reduced_subscripts: brute_force_work(&mut names, subscripts)?,
        })
    }
}

fn compute_order(ss: &[Subscripts]) -> usize {
    ss.iter()
        .map(|ss| ss.compute_order())
        .max()
        .expect("self.0 never be empty")
}

fn memory_order(ss: &[Subscripts]) -> usize {
    ss.iter()
        .map(|ss| ss.memory_order())
        .max()
        .expect("self.0 never be empty")
}

fn brute_force_work(names: &mut Namespace, subscripts: Subscripts) -> Result<Vec<Subscripts>> {
    if subscripts.inputs.len() <= 2 {
        // Cannot be factorized anymore
        return Ok(vec![subscripts]);
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
            let mut sub = brute_force_work(&mut names, outer)?;
            sub.insert(0, inner);
            Ok(sub)
        })
        .collect::<Result<Vec<_>>>()?;
    subpaths.push(vec![subscripts]);
    Ok(subpaths
        .into_iter()
        .min_by_key(|path| (compute_order(path), memory_order(path)))
        .expect("subpath never be empty"))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn brute_force_ab_bc() -> Result<()> {
        let path = Path::brute_force("ab,bc->ac")?;
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].to_string(), "ab,bc->ac | arg0,arg1->out0");
        Ok(())
    }

    #[test]
    fn brute_force_ab_bc_cd_d() -> Result<()> {
        let path = Path::brute_force("ab,bc,cd,d->a")?;
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].to_string(), "ab,b->a | arg2,arg3->out1");
        assert_eq!(path[1].to_string(), "a,ba->b | out1,arg1->out2");
        assert_eq!(path[2].to_string(), "a,ba->b | out2,arg0->out0");
        Ok(())
    }

    #[test]
    fn brute_force_a_a_a() -> Result<()> {
        let path = Path::brute_force("a,a,a->")?;
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].to_string(), "a,a,a-> | arg0,arg1,arg2->out0");
        Ok(())
    }
}
