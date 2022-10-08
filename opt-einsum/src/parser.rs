//! Parse einsum operator, e.g. `ij,jk->ik`

use nom::IResult;

/// Indices appearing in einsum operator, e.g. `ij`
pub type Indices = Vec<char>;

pub fn indices(input: &str) -> IResult<&str, Indices> {
    todo!()
}

pub struct Operator {
    inputs: Vec<Indices>,
    output: Indices,
}

pub fn parse(input: &str) -> Result<(), ()> {
    todo!()
}
