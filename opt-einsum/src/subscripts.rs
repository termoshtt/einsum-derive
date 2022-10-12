use crate::{
    error::{Error, Result},
    parser,
};
use std::str::FromStr;

/// Each subscript label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    /// Single index, e.g. `i` or `j`
    Index(char),
    /// Ellipsis `...` representing broadcast
    Ellipsis,
}

/// Each subscript appearing in einsum, e.g. `ij`
pub type SubScript = Vec<Label>;

/// Einsum subscripts, e.g. `ij,jk->ik`
#[derive(Debug, PartialEq, Eq)]
pub struct SubScripts {
    /// Input subscript, `ij` and `jk`
    pub inputs: Vec<SubScript>,
    /// Output subscript. This may be empty for "implicit mode".
    pub output: Option<SubScript>,
}

impl FromStr for SubScripts {
    type Err = Error;
    fn from_str(input: &str) -> Result<Self> {
        use nom::Finish;
        if let Ok((_, ss)) = parser::subscripts(input).finish() {
            Ok(ss)
        } else {
            Err(Error::InvalidSubScripts(input.to_string()))
        }
    }
}
