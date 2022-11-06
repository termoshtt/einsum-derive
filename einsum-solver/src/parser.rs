//! Parse einsum subscripts
//!
//! These parsers are implemented using [nom](https://github.com/Geal/nom),
//! and corresponding EBNF-like schema are written in each document page.
//!

use crate::subscripts::*;
use anyhow::{bail, Error, Result};
use nom::{
    branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*, sequence::*,
    IResult, Parser,
};

/// index = `a` | `b` | `c` | `d` | `e` | `f` | `g` | `h` | `i` | `j` | `k` | `l` |`m` | `n` | `o` | `p` | `q` | `r` | `s` | `t` | `u` | `v` | `w` | `x` |`y` | `z`;
pub fn index(input: &str) -> IResult<&str, char> {
    satisfy(|c| matches!(c, 'a'..='z')).parse(input)
}

/// ellipsis = `...`
pub fn ellipsis(input: &str) -> IResult<&str, Label> {
    tag("...").map(|_| Label::Ellipsis).parse(input)
}

/// subscript = { [index] | [ellipsis] };
pub fn subscript(input: &str) -> IResult<&str, Subscript> {
    many0(alt((
        index.map(|c| Some(Label::Index(c))),
        ellipsis.map(Some),
        multispace1.map(|_| None),
    )))
    .map(|labels| Subscript(labels.into_iter().flatten().collect()))
    .parse(input)
}

/// Einsum subscripts, e.g. `ij,jk->ik`
#[derive(Debug, PartialEq, Eq)]
pub struct RawSubscripts {
    /// Input subscript, `ij` and `jk`
    pub inputs: Vec<Subscript>,
    /// Output subscript. This may be empty for "implicit mode".
    pub output: Option<Subscript>,
}

impl std::str::FromStr for RawSubscripts {
    type Err = Error;
    fn from_str(input: &str) -> Result<Self> {
        use nom::Finish;
        if let Ok((_, ss)) = subscripts(input).finish() {
            Ok(ss)
        } else {
            bail!("Invalid subscripts: {}", input);
        }
    }
}

/// subscripts = [subscript] {`,` [subscript]} \[ `->` [subscript] \]
pub fn subscripts(input: &str) -> IResult<&str, RawSubscripts> {
    let (input, _head) = multispace0(input)?;
    let (input, inputs) = separated_list1(char(','), subscript)(input)?;
    let (input, output) = opt(tuple((multispace0, tag("->"), multispace0, subscript))
        .map(|(_space_pre, _arrow, _space_post, output)| output))(input)?;
    Ok((input, RawSubscripts { inputs, output }))
}

/// subscript_ = { [index] } [ [ellipsis] { [index] } ];
pub fn subscript_(input: &str) -> IResult<&str, Subscript_> {
    let mut indices = many0(tuple((multispace0, index)).map(|(_space, c)| c));
    let (input, start) = indices(input)?;
    let (input, end) = opt(tuple((multispace0, ellipsis, multispace0, indices))
        .map(|(_space_pre, _ellipsis, _space_post, output)| output))(input)?;
    if let Some(end) = end {
        Ok((input, Subscript_::Ellipsis { start, end }))
    } else {
        Ok((input, Subscript_::Indices(start)))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use nom::Finish;

    #[test]
    fn test_indices() {
        let ans = (
            "",
            Subscript(vec![
                Label::Index('i'),
                Label::Index('j'),
                Label::Index('k'),
            ]),
        );
        assert_eq!(subscript("ijk").finish().unwrap(), ans);
        assert_eq!(subscript("i jk").finish().unwrap(), ans);
        assert_eq!(subscript("ij k").finish().unwrap(), ans);
        assert_eq!(subscript("i j k").finish().unwrap(), ans);
    }

    #[test]
    fn test_subscript() {
        let (res, out) = subscript("i...j").finish().unwrap();
        assert_eq!(
            out,
            Subscript(vec![Label::Index('i'), Label::Ellipsis, Label::Index('j'),])
        );
        assert_eq!(res, "");

        let (res, out) = subscript("...").finish().unwrap();
        assert_eq!(out, Subscript(vec![Label::Ellipsis,]));
        assert_eq!(res, "");

        let (res, out) = subscript("...j").finish().unwrap();
        assert_eq!(out, Subscript(vec![Label::Ellipsis, Label::Index('j'),]));
        assert_eq!(res, "");
    }

    #[test]
    fn test_subscript_() {
        let (res, out) = subscript_("ijk").finish().unwrap();
        assert_eq!(out, Subscript_::Indices(vec!['i', 'j', 'k']));
        assert_eq!(res, "");

        let (res, out) = subscript_("...").finish().unwrap();
        assert_eq!(
            out,
            Subscript_::Ellipsis {
                start: Vec::new(),
                end: Vec::new()
            }
        );
        assert_eq!(res, "");

        let (res, out) = subscript_("i...").finish().unwrap();
        assert_eq!(
            out,
            Subscript_::Ellipsis {
                start: vec!['i'],
                end: Vec::new()
            }
        );
        assert_eq!(res, "");

        let (res, out) = subscript_("...j").finish().unwrap();
        assert_eq!(
            out,
            Subscript_::Ellipsis {
                start: Vec::new(),
                end: vec!['j'],
            }
        );
        assert_eq!(res, "");

        let (res, out) = subscript_("i...j").finish().unwrap();
        assert_eq!(
            out,
            Subscript_::Ellipsis {
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
                        Subscript(vec![Label::Index('i'), Label::Index('j')]),
                        Subscript(vec![Label::Index('j'), Label::Index('k')])
                    ],
                    output: Some(Subscript(vec![Label::Index('i'), Label::Index('k')])),
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
                    Subscript(vec![Label::Index('i'), Label::Index('j')]),
                    Subscript(vec![Label::Index('j'), Label::Index('k')])
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
                    Subscript(vec![Label::Index('i'), Label::Ellipsis]),
                    Subscript(vec![Label::Index('i'), Label::Ellipsis])
                ],
                output: Some(Subscript(vec![Label::Ellipsis]))
            }
        );
    }
}
