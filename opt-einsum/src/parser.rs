//! Parse einsum operator, e.g. `ij,jk->ik`

use nom::{bytes::complete::*, character::complete::*, multi::*, sequence::*, IResult};

/// Indices appearing in einsum operator, e.g. `ij`
pub type Indices<'input> = &'input str;

pub fn indices(input: &str) -> IResult<&str, Indices> {
    alpha1(input)
}

/// Einsum operator, e.g. `ij,jk->ik`
#[derive(Debug, PartialEq, Eq)]
pub struct Operator<'input> {
    /// Input indices, `ij` and `jk`
    pub inputs: Vec<Indices<'input>>,
    /// Output indices, `ik`
    pub output: Indices<'input>,
}

pub fn operator(input: &str) -> IResult<&str, Operator> {
    let (input, _head) = multispace0(input)?;
    let (input, inputs) =
        separated_list1(tuple((multispace0, char(','), multispace0)), indices)(input)?;
    let (input, _arrow) = tuple((multispace0, tag("->"), multispace0))(input)?;
    let (input, output) = indices(input)?;
    Ok((input, Operator { inputs, output }))
}

#[cfg(test)]
mod tests {

    use super::*;
    use nom::Finish;

    #[test]
    fn parse() {
        fn test(input: &str) {
            let (_, op) = operator(input).finish().unwrap();
            assert_eq!(
                op,
                Operator {
                    inputs: vec!["ij", "jk"],
                    output: "ik",
                }
            );
        }
        test("ij,jk->ik");

        // with space
        test(" ij,jk->ik");
        test("ij ,jk->ik");
        test("ij, jk->ik");
        test("ij,jk ->ik");
        test("ij,jk-> ik");
        test("ij,jk->ik ");
    }
}
