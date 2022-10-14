use crate::{
    error::{Error, Result},
    parser,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Deref,
    str::FromStr,
};

/// Each subscript label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    /// Single index, e.g. `i` or `j`
    Index(char),
    /// Ellipsis `...` representing broadcast
    Ellipsis,
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Label::Index(i) => write!(f, "{}", i),
            Label::Ellipsis => write!(f, "___"),
        }
    }
}

impl PartialEq<char> for Label {
    fn eq(&self, other: &char) -> bool {
        match self {
            Label::Index(i) => i == other,
            Label::Ellipsis => false,
        }
    }
}

/// Each subscript appearing in einsum, e.g. `ij`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subscript(pub Vec<Label>);

impl FromStr for Subscript {
    type Err = Error;
    fn from_str(input: &str) -> Result<Self> {
        if let Ok((_, ss)) = parser::subscript(input) {
            Ok(ss)
        } else {
            Err(Self::Err::InvalidSubScripts(input.to_string()))
        }
    }
}

impl Deref for Subscript {
    type Target = [Label];
    fn deref(&self) -> &[Label] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a Subscript {
    type Item = &'a Label;
    type IntoIter = std::slice::Iter<'a, Label>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<const N: usize> PartialEq<[char; N]> for Subscript {
    fn eq(&self, other: &[char; N]) -> bool {
        PartialEq::eq(&self.0, other)
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
            for label in input.as_ref() {
                write!(f, "{}", label)?;
            }
            write!(f, "_")?;
        }
        write!(f, "_")?;
        for label in self.output.as_ref() {
            write!(f, "{}", label)?;
        }
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

        let count = count_inputs(&raw.inputs);
        let output = Subscript(
            count
                .iter()
                .filter_map(|(key, value)| {
                    if *value == 1 {
                        Some(Label::Index(*key))
                    } else {
                        None
                    }
                })
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
        let count = count_inputs(&self.inputs);
        let mut subscripts: BTreeSet<char> = count
            .into_iter()
            .filter_map(|(key, value)| if value > 1 { Some(key) } else { None })
            .collect();
        for label in &self.output {
            if let Label::Index(c) = label {
                subscripts.remove(c);
            }
        }
        subscripts
    }

    /// Evaluate contracted indices
    ///
    /// ```
    /// use einsum_solver::subscripts::*;
    /// use std::str::FromStr;
    ///
    /// let subscripts = Subscripts::from_str("ij,jk,kl->il").unwrap();
    /// let contracted = subscripts.contracted('j').unwrap();
    /// assert_eq!(contracted, Subscripts::from_str("ik,kl->il").unwrap());
    /// ```
    pub fn contracted(&self, index: char) -> Result<Self> {
        if !self.contraction_indices().contains(&index) {
            return Err(Error::UnknownIndex(index));
        }

        let mut intermediate = BTreeSet::new();
        let mut others = Vec::new();
        for input in &self.inputs {
            if input.iter().any(|label| *label == index) {
                for label in input {
                    if let Label::Index(c) = label {
                        if *c != index {
                            intermediate.insert(c);
                        }
                    }
                }
            } else {
                others.push(input.clone());
            }
        }
        let mut inputs = vec![Subscript(
            intermediate
                .into_iter()
                .map(|index| Label::Index(*index))
                .collect::<Vec<Label>>(),
        )];
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

fn count_inputs(inputs: &[Subscript]) -> BTreeMap<char, u32> {
    let mut count = BTreeMap::new();
    for input in inputs {
        for label in input {
            match label {
                Label::Index(c) => count.entry(*c).and_modify(|n| *n += 1).or_insert(1),
                Label::Ellipsis => continue,
            };
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