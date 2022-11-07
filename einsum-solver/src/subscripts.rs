//! Einsum subscripts, e.g. `ij,jk->ik`
use crate::parser;
use anyhow::{bail, Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Subscript {
    /// Indices without ellipsis, e.g. `ijk`
    Indices(Vec<char>),
    /// Indices with ellipsis, e.g. `i...j`
    Ellipsis { start: Vec<char>, end: Vec<char> },
}

impl<const N: usize> PartialEq<[char; N]> for Subscript {
    fn eq(&self, other: &[char; N]) -> bool {
        match self {
            Subscript::Indices(indices) => indices.eq(other),
            _ => false,
        }
    }
}

impl Subscript {
    pub fn indices(&self) -> Vec<char> {
        match self {
            Subscript::Indices(indices) => indices.clone(),
            Subscript::Ellipsis { start, end } => start.iter().chain(end.iter()).cloned().collect(),
        }
    }
}

impl fmt::Display for Subscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Subscript::Indices(indices) => {
                for i in indices {
                    write!(f, "{}", i)?;
                }
            }
            Subscript::Ellipsis { start, end } => {
                for i in start {
                    write!(f, "{}", i)?;
                }
                write!(f, "___")?;
                for i in end {
                    write!(f, "{}", i)?;
                }
            }
        }
        Ok(())
    }
}

/// Einsum subscripts, e.g. `ij,jk->ik`
#[derive(Debug, PartialEq, Eq)]
pub struct Subscripts {
    /// Input subscript, `ij` and `jk`
    pub inputs: Vec<Subscript>,
    /// Output subscript.
    pub output: Subscript,
}

// `Display` implementation is designed to use function name.
// This is not injective, e.g. `i...,j->ij` and `i,...j->ij`
// returns a same result `i____j__ij`.
impl fmt::Display for Subscripts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for input in &self.inputs {
            write!(f, "{}", input)?;
            write!(f, "_")?;
        }
        write!(f, "_{}", self.output)?;
        Ok(())
    }
}

impl Subscripts {
    /// Normalize subscripts into "explicit mode"
    ///
    /// [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
    /// has "explicit mode" including `->`, e.g. `ij,jk->ik` and
    /// "implicit mode" e.g. `ij,jk`.
    /// The output subscript is determined from input subscripts in implicit mode:
    ///
    /// > In implicit mode, the chosen subscripts are important since the axes
    /// > of the output are reordered alphabetically.
    /// > This means that `np.einsum('ij', a)` doesn’t affect a 2D array,
    /// > while `np.einsum('ji', a)` takes its transpose.
    /// > Additionally, `np.einsum('ij,jk', a, b)` returns a matrix multiplication,
    /// > while, `np.einsum('ij,jh', a, b)` returns the transpose of
    /// > the multiplication since subscript ‘h’ precedes subscript ‘i’.
    ///
    /// ```
    /// use std::str::FromStr;
    /// use einsum_solver::{subscripts::Subscripts, parser::RawSubscripts};
    ///
    /// // Infer output subscripts for implicit mode
    /// let raw = RawSubscripts::from_str("ij,jk").unwrap();
    /// let subscripts = Subscripts::from_raw(raw);
    /// assert_eq!(subscripts.output, ['i', 'k']);
    ///
    /// // Reordered alphabetically
    /// let raw = RawSubscripts::from_str("ji").unwrap();
    /// let subscripts = Subscripts::from_raw(raw);
    /// assert_eq!(subscripts.output, ['i', 'j']);
    /// ```
    ///
    pub fn from_raw(raw: parser::RawSubscripts) -> Self {
        if let Some(output) = raw.output {
            return Subscripts {
                inputs: raw.inputs,
                output,
            };
        }

        let count = count_indices(&raw.inputs);
        let output = Subscript::Indices(
            count
                .iter()
                .filter_map(|(key, value)| if *value == 1 { Some(*key) } else { None })
                .collect(),
        );
        Subscripts {
            inputs: raw.inputs,
            output,
        }
    }

    /// Indices to be contracted
    ///
    /// ```
    /// use std::str::FromStr;
    /// use maplit::btreeset;
    /// use einsum_solver::subscripts::Subscripts;
    ///
    /// // Matrix multiplication AB
    /// let subscripts = Subscripts::from_str("ij,jk->ik").unwrap();
    /// assert_eq!(subscripts.contraction_indices(), btreeset!{'j'});
    ///
    /// // Reduce all Tr(AB)
    /// let subscripts = Subscripts::from_str("ij,ji->").unwrap();
    /// assert_eq!(subscripts.contraction_indices(), btreeset!{'i', 'j'});
    ///
    /// // Take diagonal elements
    /// let subscripts = Subscripts::from_str("ii->i").unwrap();
    /// assert_eq!(subscripts.contraction_indices(), btreeset!{});
    /// ```
    pub fn contraction_indices(&self) -> BTreeSet<char> {
        let count = count_indices(&self.inputs);
        let mut subscripts: BTreeSet<char> = count
            .into_iter()
            .filter_map(|(key, value)| if value > 1 { Some(key) } else { None })
            .collect();
        for c in &self.output.indices() {
            subscripts.remove(c);
        }
        subscripts
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// Evaluate contracted indices
    ///
    /// The memorized storage is placed on the first of input.
    /// For example, the indices of memorized storage of $j$-contraction
    /// for `ij,jk,kl->il` will be `ik`,
    /// and this function yields subscript `ik,kl->il` instead of `kl,ik->il`.
    ///
    /// ```
    /// use einsum_solver::subscripts::*;
    /// use std::str::FromStr;
    ///
    /// let base = Subscripts::from_str("ij,jk,kl->il").unwrap();
    ///
    /// // j -> k
    /// let j_contracted = base.contracted('j').unwrap();
    /// assert_eq!(j_contracted, Subscripts::from_str("ik,kl->il").unwrap());
    /// let jk_contracted = j_contracted.contracted('k').unwrap();
    /// assert_eq!(jk_contracted, Subscripts::from_str("il->il").unwrap());
    ///
    /// // k -> j
    /// let k_contracted = base.contracted('k').unwrap();
    /// assert_eq!(k_contracted, Subscripts::from_str("jl,ij->il").unwrap());
    /// let kj_contracted = k_contracted.contracted('j').unwrap();
    /// assert_eq!(kj_contracted, Subscripts::from_str("il->il").unwrap());
    /// ```
    pub fn contracted(&self, index: char) -> Result<Self> {
        if !self.contraction_indices().contains(&index) {
            bail!("Unknown index: {}", index);
        }

        let mut intermediate = BTreeSet::new();
        let mut others = Vec::new();
        for input in &self.inputs {
            let indices = input.indices();
            if indices.iter().any(|label| *label == index) {
                for c in indices {
                    if c != index {
                        intermediate.insert(c);
                    }
                }
            } else {
                others.push(input.clone());
            }
        }
        let mut inputs = vec![Subscript::Indices(intermediate.into_iter().collect())];
        for other in others {
            inputs.push(other)
        }
        Ok(Self {
            inputs,
            output: self.output.clone(),
        })
    }
}

impl FromStr for Subscripts {
    type Err = Error;
    fn from_str(input: &str) -> Result<Self> {
        let raw = parser::RawSubscripts::from_str(input)?;
        Ok(Self::from_raw(raw))
    }
}

impl From<parser::RawSubscripts> for Subscripts {
    fn from(raw: parser::RawSubscripts) -> Self {
        Self::from_raw(raw)
    }
}

fn count_indices(inputs: &[Subscript]) -> BTreeMap<char, u32> {
    let mut count = BTreeMap::new();
    for input in inputs {
        match input {
            Subscript::Indices(indices) => {
                for c in indices {
                    count.entry(*c).and_modify(|n| *n += 1).or_insert(1);
                }
            }
            Subscript::Ellipsis { start, end } => {
                for c in start.iter().chain(end.iter()) {
                    count.entry(*c).and_modify(|n| *n += 1).or_insert(1);
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let subscripts = Subscripts::from_str("ij,jk->ik").unwrap();
        assert_eq!(format!("{}", subscripts), "ij_jk__ik");

        // implicit mode
        let subscripts = Subscripts::from_str("ij,jk").unwrap();
        assert_eq!(format!("{}", subscripts), "ij_jk__ik");

        // output scalar
        let subscripts = Subscripts::from_str("i,i").unwrap();
        assert_eq!(format!("{}", subscripts), "i_i__");

        // ellipsis
        let subscripts = Subscripts::from_str("ij...,jk...->ik...").unwrap();
        assert_eq!(format!("{}", subscripts), "ij____jk_____ik___");
    }
}
