use crate::{
    error::{Error, Result},
    parser,
};
use std::{collections::BTreeSet, str::FromStr};

/// Each subscript label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    /// Single index, e.g. `i` or `j`
    Index(char),
    /// Ellipsis `...` representing broadcast
    Ellipsis,
}

/// Each subscript appearing in einsum, e.g. `ij`
pub type Subscript = Vec<Label>;

/// Einsum subscripts, e.g. `ij,jk->ik`
#[derive(Debug, PartialEq, Eq)]
pub struct Subscripts {
    /// Input subscript, `ij` and `jk`
    pub inputs: Vec<Subscript>,
    /// Output subscript.
    pub output: Subscript,
}

impl Subscripts {
    pub fn from_raw(_raw: parser::RawSubscripts) -> Result<Self> {
        todo!()
    }

    /// Subscripts to be contracted
    ///
    /// ```
    /// use std::str::FromStr;
    /// use opt_einsum::subscripts::Subscripts;
    ///
    /// // Matrix multiplication AB
    /// let subscripts = Subscripts::from_str("ij,jk->ik").unwrap();
    /// assert_eq!(subscripts.contraction_subscripts(), vec!['j']);
    ///
    /// // Reduce all Tr(AB)
    /// let subscripts = Subscripts::from_str("ij,ji->").unwrap();
    /// assert_eq!(subscripts.contraction_subscripts(), vec!['i', 'j']);
    ///
    /// // Take diagonal elements
    /// let subscripts = Subscripts::from_str("ii->i").unwrap();
    /// assert_eq!(subscripts.contraction_subscripts(), vec![]);
    /// ```
    pub fn contraction_subscripts(&self) -> BTreeSet<char> {
        todo!()
    }
}

impl FromStr for Subscripts {
    type Err = Error;
    fn from_str(input: &str) -> Result<Self> {
        use nom::Finish;
        if let Ok((_, ss)) = parser::subscripts(input).finish() {
            Self::from_raw(ss)
        } else {
            Err(Error::InvalidSubScripts(input.to_string()))
        }
    }
}
