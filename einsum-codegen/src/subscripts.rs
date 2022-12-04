//! Einsum subscripts, e.g. `ij,jk->ik`
use crate::{parser::*, *};
use anyhow::{bail, Error, Result};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RawSubscript {
    /// Indices without ellipsis, e.g. `ijk`
    Indices(Vec<char>),
    /// Indices with ellipsis, e.g. `i...j`
    Ellipsis { start: Vec<char>, end: Vec<char> },
}

impl<const N: usize> PartialEq<[char; N]> for RawSubscript {
    fn eq(&self, other: &[char; N]) -> bool {
        match self {
            RawSubscript::Indices(indices) => indices.eq(other),
            _ => false,
        }
    }
}

impl fmt::Display for RawSubscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RawSubscript::Indices(indices) => {
                for i in indices {
                    write!(f, "{}", i)?;
                }
            }
            RawSubscript::Ellipsis { start, end } => {
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
pub struct RawSubscripts {
    /// Input subscript, `ij` and `jk`
    pub inputs: Vec<RawSubscript>,
    /// Output subscript. This may be empty for "implicit mode".
    pub output: Option<RawSubscript>,
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

/// Subscripts corresponding to DOT in BLAS
pub fn dot() -> RawSubscripts {
    RawSubscripts::from_str("a,a->").unwrap()
}

/// Subscripts corresponding to GEMM in BLAS
pub fn gemm() -> RawSubscripts {
    RawSubscripts::from_str("ab,bc->ac").unwrap()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subscript {
    raw: RawSubscript,
    position: Position,
}

impl Subscript {
    pub fn raw(&self) -> &RawSubscript {
        &self.raw
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn indices(&self) -> Vec<char> {
        match &self.raw {
            RawSubscript::Indices(indices) => indices.clone(),
            RawSubscript::Ellipsis { start, end } => {
                start.iter().chain(end.iter()).cloned().collect()
            }
        }
    }
}

impl ToTokens for Subscript {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        ToTokens::to_tokens(&self.position, tokens)
    }
}

#[cfg_attr(doc, katexit::katexit)]
/// Einsum subscripts with tensor names, e.g. `ab,bc->ac | arg0,arg1->out0`
///
/// Indices are remapped as starting from `a` to distinguish same subscripts, e.g. `i,i->` and `j,j->`
///
/// ```
/// use einsum_codegen::*;
///
/// let mut names = Namespace::init();
/// let mut ss1 = Subscripts::from_raw_indices(&mut names, "ij,jk,kl->il").unwrap();
///
/// let mut names = Namespace::init();
/// let mut ss2 = Subscripts::from_raw_indices(&mut names, "xz,zy,yw->xw").unwrap();
///
/// assert_eq!(ss1, ss2);
/// assert_eq!(ss1.to_string(), "ab,bc,cd->ad | arg0,arg1,arg2->out0");
/// assert_eq!(ss2.to_string(), "ab,bc,cd->ad | arg0,arg1,arg2->out0");
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Subscripts {
    /// Input subscript, `ij` and `jk`
    pub inputs: Vec<Subscript>,
    /// Output subscript.
    pub output: Subscript,
}

// `ij,jk->ik | arg0,arg1->out0` format
impl fmt::Debug for Subscripts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (n, input) in self.inputs.iter().enumerate() {
            write!(f, "{}", input.raw)?;
            if n < self.inputs.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "->{} | ", self.output.raw)?;

        for (n, input) in self.inputs.iter().enumerate() {
            write!(f, "{}", input.position)?;
            if n < self.inputs.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "->{}", self.output.position)?;
        Ok(())
    }
}

impl fmt::Display for Subscripts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl ToTokens for Subscripts {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let fn_name = format_ident!("{}", self.escaped_ident());
        let args = &self.inputs;
        let out = &self.output;
        tokens.append_all(quote! {
            let #out = #fn_name(#(#args),*);
        });
    }
}

impl Subscripts {
    /// Returns $\alpha$ if this subscripts requires $O(N^\alpha)$ floating point operation
    pub fn compute_order(&self) -> usize {
        self.memory_order() + self.contraction_indices().len()
    }

    /// Returns $\beta$ if this subscripts requires $O(N^\beta)$ memory
    pub fn memory_order(&self) -> usize {
        self.output.indices().len()
    }

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
    /// use einsum_codegen::*;
    ///
    /// // Infer output subscripts for implicit mode
    /// let mut names = Namespace::init();
    /// let raw = RawSubscripts::from_str("ab,bc").unwrap();
    /// let subscripts = Subscripts::from_raw(&mut names, raw);
    /// assert_eq!(subscripts.to_string(), "ab,bc->ac | arg0,arg1->out0");
    ///
    /// // Reordered alphabetically
    /// let mut names = Namespace::init(); // reset namespace
    /// let raw = RawSubscripts::from_str("ba").unwrap();
    /// let subscripts = Subscripts::from_raw(&mut names, raw);
    /// assert_eq!(subscripts.to_string(), "ab->ba | arg0->out0");
    /// ```
    ///
    pub fn from_raw(names: &mut Namespace, raw: RawSubscripts) -> Self {
        let inputs = raw
            .inputs
            .iter()
            .enumerate()
            .map(|(i, indices)| Subscript {
                raw: indices.clone(),
                position: Position::Arg(i),
            })
            .collect();
        let position = names.new_ident();
        if let Some(output) = raw.output {
            let mut cand = Subscripts {
                inputs,
                output: Subscript {
                    raw: output,
                    position,
                },
            };
            cand.remap_indices();
            return cand;
        }

        let count = count_indices(&inputs);
        let output = Subscript {
            raw: RawSubscript::Indices(
                count
                    .iter()
                    .filter_map(|(key, value)| if *value == 1 { Some(*key) } else { None })
                    .collect(),
            ),
            position,
        };
        let mut cand = Subscripts { inputs, output };
        cand.remap_indices();
        cand
    }

    pub fn from_raw_indices(names: &mut Namespace, indices: &str) -> Result<Self> {
        let raw = RawSubscripts::from_str(indices)?;
        Ok(Self::from_raw(names, raw))
    }

    /// Indices to be contracted
    ///
    /// ```
    /// use std::str::FromStr;
    /// use maplit::btreeset;
    /// use einsum_codegen::*;
    ///
    /// let mut names = Namespace::init();
    ///
    /// // Matrix multiplication AB
    /// let subscripts = Subscripts::from_raw_indices(&mut names, "ab,bc->ac").unwrap();
    /// assert_eq!(subscripts.contraction_indices(), btreeset!{'b'});
    ///
    /// // Reduce all Tr(AB)
    /// let subscripts = Subscripts::from_raw_indices(&mut names, "ab,ba->").unwrap();
    /// assert_eq!(subscripts.contraction_indices(), btreeset!{'a', 'b'});
    ///
    /// // Take diagonal elements
    /// let subscripts = Subscripts::from_raw_indices(&mut names, "aa->a").unwrap();
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

    /// Factorize subscripts
    ///
    /// ```text
    /// ab,bc,cd->ad | arg0,arg1,arg2->out0
    /// ```
    ///
    /// will be factorized with `(arg0, arg1)` into
    ///
    /// ```text
    /// ab,bc->ac | arg0,arg1 -> out1
    /// ab,bc->ac | out1 arg2 -> out0
    /// ```
    ///
    /// Be sure that the indices of `out1` in the first step `ac` is renamed
    /// into `ab` in the second step.
    ///
    /// ```
    /// use einsum_codegen::*;
    /// use std::str::FromStr;
    /// use maplit::btreeset;
    ///
    /// let mut names = Namespace::init();
    /// let base = Subscripts::from_raw_indices(&mut names, "ab,bc,cd->ad").unwrap();
    ///
    /// let (step1, step2) = base.factorize(&mut names,
    ///   btreeset!{ Position::Arg(0), Position::Arg(1) }
    /// ).unwrap();
    ///
    /// assert_eq!(step1.to_string(), "ab,bc->ac | arg0,arg1->out1");
    /// assert_eq!(step2.to_string(), "ab,bc->ac | out1,arg2->out0");
    /// ```
    pub fn factorize(
        &self,
        names: &mut Namespace,
        inners: BTreeSet<Position>,
    ) -> Result<(Self, Self)> {
        let mut inner_inputs = Vec::new();
        let mut outer_inputs = Vec::new();
        let mut indices: BTreeMap<char, (usize /* inner */, usize /* outer */)> = BTreeMap::new();
        for input in &self.inputs {
            if inners.contains(&input.position) {
                inner_inputs.push(input.clone());
                for c in input.indices() {
                    indices
                        .entry(c)
                        .and_modify(|(i, _)| *i += 1)
                        .or_insert((1, 0));
                }
            } else {
                outer_inputs.push(input.clone());
                for c in input.indices() {
                    indices
                        .entry(c)
                        .and_modify(|(_, o)| *o += 1)
                        .or_insert((0, 1));
                }
            }
        }
        let out = Subscript {
            raw: RawSubscript::Indices(
                indices
                    .into_iter()
                    .filter_map(|(key, (i, o))| {
                        if i == 1 || (i >= 2 && o > 0) {
                            Some(key)
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
            position: names.new_ident(),
        };
        outer_inputs.insert(0, out.clone());

        let mut inner = Subscripts {
            inputs: inner_inputs,
            output: out,
        };
        let mut outer = Subscripts {
            inputs: outer_inputs,
            output: self.output.clone(),
        };
        inner.remap_indices();
        outer.remap_indices();
        Ok((inner, outer))
    }

    /// Escaped subscript for identifier
    ///
    /// This is not injective, e.g. `i...,j->ij` and `i,...j->ij`
    /// returns a same result `i____j__ij`.
    ///
    pub fn escaped_ident(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for input in &self.inputs {
            write!(out, "{}", input.raw).unwrap();
            write!(out, "_").unwrap();
        }
        write!(out, "_{}", self.output.raw).unwrap();
        out
    }

    fn remap_indices(&mut self) {
        let mut map: BTreeMap<char, u32> = BTreeMap::new();
        let mut update = |raw: &mut RawSubscript| match raw {
            RawSubscript::Indices(indices) => {
                for i in indices {
                    if !map.contains_key(i) {
                        map.insert(*i, 'a' as u32 + map.len() as u32);
                    }
                    *i = char::from_u32(map[i]).unwrap();
                }
            }
            RawSubscript::Ellipsis { start, end } => {
                for i in start.iter_mut().chain(end.iter_mut()) {
                    if !map.contains_key(i) {
                        map.insert(*i, 'a' as u32 + map.len() as u32);
                    }
                    *i = char::from_u32(map[i]).unwrap();
                }
            }
        };
        for input in &mut self.inputs {
            update(&mut input.raw);
        }
        update(&mut self.output.raw)
    }
}

fn count_indices(inputs: &[Subscript]) -> BTreeMap<char, u32> {
    let mut count = BTreeMap::new();
    for input in inputs {
        for c in input.indices() {
            count.entry(c).and_modify(|n| *n += 1).or_insert(1);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaped_ident() {
        let mut names = Namespace::init();

        let subscripts = Subscripts::from_raw_indices(&mut names, "ab,bc->ac").unwrap();
        assert_eq!(subscripts.escaped_ident(), "ab_bc__ac");

        // implicit mode
        let subscripts = Subscripts::from_raw_indices(&mut names, "ab,bc").unwrap();
        assert_eq!(subscripts.escaped_ident(), "ab_bc__ac");

        // output scalar
        let subscripts = Subscripts::from_raw_indices(&mut names, "a,a").unwrap();
        assert_eq!(subscripts.escaped_ident(), "a_a__");

        // ellipsis
        let subscripts = Subscripts::from_raw_indices(&mut names, "ab...,bc...->ac...").unwrap();
        assert_eq!(subscripts.escaped_ident(), "ab____bc_____ac___");
    }
}
