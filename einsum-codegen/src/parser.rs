//! Parse einsum subscripts
//!
//! These parsers are implemented using [nom](https://github.com/Geal/nom),
//! and corresponding EBNF-like schema are written in each document page.
//!

use crate::{RawSubscript, RawSubscripts};
use nom::{
    bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*, IResult,
    Parser,
};

/// index = `a` | `b` | `c` | `d` | `e` | `f` | `g` | `h` | `i` | `j` | `k` | `l` |`m` | `n` | `o` | `p` | `q` | `r` | `s` | `t` | `u` | `v` | `w` | `x` |`y` | `z`;
pub fn index(input: &str) -> IResult<&str, char> {
    satisfy(|c| c.is_ascii_lowercase()).parse(input)
}

/// ellipsis = `...`
pub fn ellipsis(input: &str) -> IResult<&str, &str> {
    tag("...").parse(input)
}

/// subscript = { [index] } [ [ellipsis] { [index] } ];
pub fn subscript(input: &str) -> IResult<&str, RawSubscript> {
    let mut indices = many0(tuple((multispace0, index)).map(|(_space, c)| c));
    let (input, start) = indices(input)?;
    let (input, end) = opt(tuple((multispace0, ellipsis, multispace0, indices))
        .map(|(_space_pre, _ellipsis, _space_post, output)| output))(input)?;
    if let Some(end) = end {
        Ok((input, RawSubscript::Ellipsis { start, end }))
    } else {
        Ok((input, RawSubscript::Indices(start)))
    }
}

/// subscripts = [subscript] {`,` [subscript]} \[ `->` [subscript] \]
pub fn subscripts(input: &str) -> IResult<&str, RawSubscripts> {
    let (input, _head) = multispace0(input)?;
    let (input, inputs) = separated_list1(tuple((multispace0, char(','))), subscript)(input)?;
    let (input, output) = opt(tuple((multispace0, tag("->"), multispace0, subscript))
        .map(|(_space_pre, _arrow, _space_post, output)| output))(input)?;
    Ok((input, RawSubscripts { inputs, output }))
}

#[cfg(test)]
mod tests {

    use super::*;
    use nom::Finish;

    #[test]
    fn test_subscript() {
        let (res, out) = subscript("ijk").finish().unwrap();
        assert_eq!(out, RawSubscript::Indices(vec!['i', 'j', 'k']));
        assert_eq!(res, "");

        let (res, out) = subscript("...").finish().unwrap();
        assert_eq!(
            out,
            RawSubscript::Ellipsis {
                start: Vec::new(),
                end: Vec::new()
            }
        );
        assert_eq!(res, "");

        let (res, out) = subscript("i...").finish().unwrap();
        assert_eq!(
            out,
            RawSubscript::Ellipsis {
                start: vec!['i'],
                end: Vec::new()
            }
        );
        assert_eq!(res, "");

        let (res, out) = subscript("...j").finish().unwrap();
        assert_eq!(
            out,
            RawSubscript::Ellipsis {
                start: Vec::new(),
                end: vec!['j'],
            }
        );
        assert_eq!(res, "");

        let (res, out) = subscript("i...j").finish().unwrap();
        assert_eq!(
            out,
            RawSubscript::Ellipsis {
                start: vec!['i'],
                end: vec!['j'],
            }
        );
        assert_eq!(res, "");
    }

    #[test]
    fn test_operator() {
        fn test(input: &str) {
            dbg!(input);
            let (_, op) = subscripts(input).finish().unwrap();
            assert_eq!(
                op,
                RawSubscripts {
                    inputs: vec![
                        RawSubscript::Indices(vec!['i', 'j']),
                        RawSubscript::Indices(vec!['j', 'k'])
                    ],
                    output: Some(RawSubscript::Indices(vec!['i', 'k'])),
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
            RawSubscripts {
                inputs: vec![
                    RawSubscript::Indices(vec!['i', 'j']),
                    RawSubscript::Indices(vec!['j', 'k'])
                ],
                output: None,
            }
        );

        // with ...
        let (_, op) = subscripts("i...,i...->...").finish().unwrap();
        assert_eq!(
            op,
            RawSubscripts {
                inputs: vec![
                    RawSubscript::Ellipsis {
                        start: vec!['i'],
                        end: Vec::new()
                    },
                    RawSubscript::Ellipsis {
                        start: vec!['i'],
                        end: Vec::new()
                    }
                ],
                output: Some(RawSubscript::Ellipsis {
                    start: Vec::new(),
                    end: Vec::new()
                })
            }
        );
    }
}
