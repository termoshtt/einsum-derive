//! Parse einsum subscripts, e.g. `ij,jk->ik`
//!
//! These parsers are implemented using [nom](https://github.com/Geal/nom),
//! and corresponding EBNF-like schema are written in each document page.
//!

use crate::subscripts::*;
use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*,
    IResult, Parser,
};

/// index = `a` | `b` | `c` | `d` | `e` | `f` | `g` | `h` | `i` | `j` | `k` | `l` |`m` | `n` | `o` | `p` | `q` | `r` | `s` | `t` | `u` | `v` | `w` | `x` |`y` | `z`;
pub fn index(input: &str) -> IResult<&str, Label> {
    satisfy(|c| matches!(c, 'a'..='z'))
        .map(Label::Index)
        .parse(input)
}

/// ellipsis = `...`
pub fn ellipsis(input: &str) -> IResult<&str, Label> {
    tag("...").map(|_| Label::Ellipsis).parse(input)
}

/// Each subscript appearing in einsum, e.g. `ij`
pub type SubScript = Vec<Label>;

/// subscript = { [index] | [ellipsis] };
pub fn subscript(input: &str) -> IResult<&str, SubScript> {
    many0(alt((index.map(Some), multispace1.map(|_| None))))
        .map(|chars| chars.into_iter().flatten().collect())
        .parse(input)
}

/// subscripts = [subscript] {`,` [subscript]} \[ `->` [subscript] \]
pub fn subscripts(input: &str) -> IResult<&str, SubScripts> {
    let (input, _head) = multispace0(input)?;
    let (input, inputs) = separated_list1(char(','), subscript)(input)?;
    let (input, output) = opt(tuple((multispace0, tag("->"), multispace0, subscript))
        .map(|(_space_pre, _arrow, _space_post, output)| output))(input)?;
    Ok((input, SubScripts { inputs, output }))
}

#[cfg(test)]
mod tests {

    use super::*;
    use nom::Finish;

    #[test]
    fn test_indices() {
        let ans = (
            "",
            vec![Label::Index('i'), Label::Index('j'), Label::Index('k')],
        );
        assert_eq!(subscript("ijk").finish().unwrap(), ans);
        assert_eq!(subscript("i jk").finish().unwrap(), ans);
        assert_eq!(subscript("ij k").finish().unwrap(), ans);
        assert_eq!(subscript("i j k").finish().unwrap(), ans);
    }

    #[test]
    fn test_operator() {
        fn test(input: &str) {
            dbg!(input);
            let (_, op) = subscripts(input).finish().unwrap();
            assert_eq!(
                op,
                SubScripts {
                    inputs: vec![
                        vec![Label::Index('i'), Label::Index('j')],
                        vec![Label::Index('j'), Label::Index('k')]
                    ],
                    output: Some(vec![Label::Index('i'), Label::Index('k')]),
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

        // implicit mode
        let (_, op) = subscripts("ij,jk").finish().unwrap();
        assert_eq!(
            op,
            SubScripts {
                inputs: vec![
                    vec![Label::Index('i'), Label::Index('j')],
                    vec![Label::Index('j'), Label::Index('k')]
                ],
                output: None,
            }
        );
    }
}
