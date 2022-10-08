//! Parse einsum operator, e.g. `ij,jk->ik`

use nom::{
    branch::*, bytes::complete::*, character::complete::*, multi::*, sequence::*, IResult, Parser,
};

pub fn index(input: &str) -> IResult<&str, char> {
    satisfy(|c| matches!(c, 'a'..='z'))(input)
}

/// Indices appearing in einsum operator, e.g. `ij`
pub type Indices = Vec<char>;

pub fn indices(input: &str) -> IResult<&str, Indices> {
    many0(alt((index.map(|c| Some(c)), multispace1.map(|_| None))))
        .map(|chars| chars.into_iter().flatten().collect())
        .parse(input)
}

/// Einsum operator, e.g. `ij,jk->ik`
#[derive(Debug, PartialEq, Eq)]
pub struct Operator {
    /// Input indices, `ij` and `jk`
    pub inputs: Vec<Indices>,
    /// Output indices, `ik`
    pub output: Indices,
}

pub fn operator(input: &str) -> IResult<&str, Operator> {
    let (input, _head) = multispace0(input)?;
    let (input, inputs) = separated_list1(char(','), indices)(input)?;
    let (input, _arrow) = tuple((multispace0, tag("->"), multispace0))(input)?;
    let (input, output) = indices(input)?;
    Ok((input, Operator { inputs, output }))
}

#[cfg(test)]
mod tests {

    use super::*;
    use nom::Finish;

    #[test]
    fn test_indices() {
        let ans = ("", vec!['i', 'j', 'k']);
        assert_eq!(indices("ijk").finish().unwrap(), ans);
        assert_eq!(indices("i jk").finish().unwrap(), ans);
        assert_eq!(indices("ij k").finish().unwrap(), ans);
        assert_eq!(indices("i j k").finish().unwrap(), ans);
    }

    #[test]
    fn test_operator() {
        fn test(input: &str) {
            dbg!(input);
            let (_, op) = operator(input).finish().unwrap();
            assert_eq!(
                op,
                Operator {
                    inputs: vec![vec!['i', 'j'], vec!['j', 'k']],
                    output: vec!['i', 'k'],
                }
            );
        }
        test("ij,jk->ik");

        // with space
        test(" ij,jk->ik");
        test("i j,jk->ik");
        test("ij ,jk->ik");
        test("ij, jk->ik");
        test("ij,j k->ik");
        test("ij,jk ->ik");
        test("ij,jk-> ik");
        test("ij,jk->i k");
    }
}
