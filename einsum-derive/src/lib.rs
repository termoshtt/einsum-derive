//! proc-macro based einsum implementation
//!
//! ```
//! use ndarray::array;
//! use einsum_derive::einsum;
//!
//! let a = array![
//!   [1.0, 2.0],
//!   [3.0, 4.0]
//! ];
//! let b = array![
//!   [1.0, 2.0],
//!   [3.0, 4.0]
//! ];
//! let c = einsum!("ij,jk->ik", a, b);
//! assert_eq!(c, array![
//!   [6.0, 8.0],
//!   [12.0, 16.0]
//! ]);
//! ```
//!
//! This proc-macro wil compile the input subscripts `"ij,jk->ik"`
//! to generate Rust code executing corresponding operation.
//!
//! Examples
//! ---------
//!
//! - `matmul3`
//!
//!   ```
//!   use ndarray::array;
//!   use einsum_derive::einsum;
//!
//!   let a = array![[1.0, 2.0], [3.0, 4.0]];
//!   let b = array![[1.0, 2.0], [3.0, 4.0]];
//!   let c = array![[1.0, 2.0], [3.0, 4.0]];
//!   let d = einsum!("ij,jk,kl->il", a, b, c);
//!   assert_eq!(d, array![[24.0, 32.0], [48.0, 64.0]]);
//!   ```
//!
//! - Take diagonal elements
//!
//!   ```
//!   use ndarray::array;
//!   use einsum_derive::einsum;
//!
//!   let a = array![[1.0, 2.0], [3.0, 4.0]];
//!   let d = einsum!("ii->i", a);
//!   assert_eq!(d, array![1.0, 4.0]);
//!   ```
//!
//! - If the subscripts and the number of input mismatches,
//!   this raises compile error:
//!
//!   ```compile_fail
//!   use ndarray::array;
//!   use einsum_derive::einsum;
//!
//!   let a = array![
//!     [1.0, 2.0],
//!     [3.0, 4.0]
//!   ];
//!   let c = einsum!("ij,jk->ik", a /* needs one more arg */);
//!   ```
//!

use einsum_codegen::{codegen::ndarray::*, *};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::{abort_call_site, proc_macro_error};
use quote::quote;
use syn::parse::Parser;

/// proc-macro based einsum
#[proc_macro_error]
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    einsum2(input.into()).into()
}

fn einsum2(input: TokenStream2) -> TokenStream2 {
    let (subscripts, args) = parse(input);
    let arg_ident: Vec<_> = (0..args.len()).map(Position::Arg).collect();
    let path = Path::brute_force(&subscripts).expect("Failed to construct execution path");
    let fn_defs: Vec<_> = path
        .iter()
        .map(|ss| {
            let inner = naive::inner(ss);
            function_definition(ss, inner)
        })
        .collect();
    let out = path.output();
    if path.num_args() != args.len() {
        abort_call_site!(
            "Argument number mismatch: subscripts ({}), args ({})",
            path.num_args(),
            args.len()
        )
    }

    quote! {
        {
            #(#fn_defs)*
            #(let #arg_ident = #args;)*
            #(#path)*
            #out
        }
    }
}

fn parse(input: TokenStream2) -> (String, Vec<syn::Expr>) {
    let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
    let args = parser.parse2(input).expect("Invalid input for einsum!");
    let mut iter = args.into_iter();
    let subscripts = if let Some(syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit),
        attrs: _,
    })) = iter.next()
    {
        lit.value()
    } else {
        panic!("einsum! must start with subscript string literal")
    };
    let args = iter.collect::<Vec<_>>();
    (subscripts, args)
}

#[cfg(test)]
mod test {
    use super::*;
    use einsum_codegen::codegen::format_block;
    use std::str::FromStr;

    #[test]
    fn test_parse() {
        let input = TokenStream2::from_str(r#""ij,jk->ik", a, b"#).unwrap();
        let (subscripts, exprs) = parse(input);
        assert_eq!(subscripts, "ij,jk->ik");
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0], syn::parse_str::<syn::Expr>("a").unwrap());
        assert_eq!(exprs[1], syn::parse_str::<syn::Expr>("b").unwrap());
    }

    #[test]
    fn einsum_ij_jk() {
        let input = TokenStream2::from_str(r#""ij,jk->ik", a, b"#).unwrap();
        let tt = format_block(einsum2(input).to_string());
        insta::assert_snapshot!(tt, @r###"
        {
            fn ij_jk__ik<T, S0, S1>(
                arg0: ndarray::ArrayBase<S0, ndarray::Ix2>,
                arg1: ndarray::ArrayBase<S1, ndarray::Ix2>,
            ) -> ndarray::Array<T, ndarray::Ix2>
            where
                T: ndarray::LinalgScalar,
                S0: ndarray::Data<Elem = T>,
                S1: ndarray::Data<Elem = T>,
            {
                let (n_i, n_j) = arg0.dim();
                let (_, n_k) = arg1.dim();
                {
                    let (n_0, n_1) = arg0.dim();
                    assert_eq!(n_0, n_i);
                    assert_eq!(n_1, n_j);
                }
                {
                    let (n_0, n_1) = arg1.dim();
                    assert_eq!(n_0, n_j);
                    assert_eq!(n_1, n_k);
                }
                let mut out0 = ndarray::Array::zeros((n_i, n_k));
                for i in 0..n_i {
                    for k in 0..n_k {
                        for j in 0..n_j {
                            out0[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
                        }
                    }
                }
                out0
            }
            let arg0 = a;
            let arg1 = b;
            let out0 = ij_jk__ik(arg0, arg1);
            out0
        }
        "###);
    }

    #[test]
    fn einsum_ij_jk_kl() {
        let input = TokenStream2::from_str(r#""ij,jk,kl->il", a, b, c"#).unwrap();
        let tt = format_block(einsum2(input).to_string());
        insta::assert_snapshot!(tt, @r###"
        {
            fn ij_jk__ik<T, S0, S1>(
                arg0: ndarray::ArrayBase<S0, ndarray::Ix2>,
                arg1: ndarray::ArrayBase<S1, ndarray::Ix2>,
            ) -> ndarray::Array<T, ndarray::Ix2>
            where
                T: ndarray::LinalgScalar,
                S0: ndarray::Data<Elem = T>,
                S1: ndarray::Data<Elem = T>,
            {
                let (n_i, n_j) = arg0.dim();
                let (_, n_k) = arg1.dim();
                {
                    let (n_0, n_1) = arg0.dim();
                    assert_eq!(n_0, n_i);
                    assert_eq!(n_1, n_j);
                }
                {
                    let (n_0, n_1) = arg1.dim();
                    assert_eq!(n_0, n_j);
                    assert_eq!(n_1, n_k);
                }
                let mut out1 = ndarray::Array::zeros((n_i, n_k));
                for i in 0..n_i {
                    for k in 0..n_k {
                        for j in 0..n_j {
                            out1[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
                        }
                    }
                }
                out1
            }
            fn ik_kl__il<T, S0, S1>(
                out1: ndarray::ArrayBase<S0, ndarray::Ix2>,
                arg2: ndarray::ArrayBase<S1, ndarray::Ix2>,
            ) -> ndarray::Array<T, ndarray::Ix2>
            where
                T: ndarray::LinalgScalar,
                S0: ndarray::Data<Elem = T>,
                S1: ndarray::Data<Elem = T>,
            {
                let (n_i, n_k) = out1.dim();
                let (_, n_l) = arg2.dim();
                {
                    let (n_0, n_1) = out1.dim();
                    assert_eq!(n_0, n_i);
                    assert_eq!(n_1, n_k);
                }
                {
                    let (n_0, n_1) = arg2.dim();
                    assert_eq!(n_0, n_k);
                    assert_eq!(n_1, n_l);
                }
                let mut out0 = ndarray::Array::zeros((n_i, n_l));
                for i in 0..n_i {
                    for l in 0..n_l {
                        for k in 0..n_k {
                            out0[(i, l)] = out1[(i, k)] * arg2[(k, l)];
                        }
                    }
                }
                out0
            }
            let arg0 = a;
            let arg1 = b;
            let arg2 = c;
            let out1 = ij_jk__ik(arg0, arg1);
            let out0 = ik_kl__il(out1, arg2);
            out0
        }
        "###);
    }
}
