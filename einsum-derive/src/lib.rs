#![doc = include_str!("../README.md")]

use einsum_codegen::{codegen::ndarray::*, *};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::{abort_call_site, proc_macro_error};
use quote::quote;
use std::collections::BTreeSet;
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
    let mut defined = BTreeSet::new();
    let fn_defs: Vec<_> = path
        .iter()
        .filter_map(|ss| {
            if defined.contains(&ss.escaped_ident()) {
                None
            } else {
                defined.insert(ss.escaped_ident());
                let inner = naive::inner(ss);
                Some(function_definition(ss, inner))
            }
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
        let input = TokenStream2::from_str(r#""ab,bc->ac", x, y"#).unwrap();
        let (subscripts, exprs) = parse(input);
        assert_eq!(subscripts, "ab,bc->ac");
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0], syn::parse_str::<syn::Expr>("x").unwrap());
        assert_eq!(exprs[1], syn::parse_str::<syn::Expr>("y").unwrap());
    }

    #[test]
    fn einsum_ab_bc() {
        let input = TokenStream2::from_str(r#""ab,bc->ac", x, y"#).unwrap();
        let tt = format_block(einsum2(input).to_string());
        insta::assert_snapshot!(tt, @r###"
        {
            fn ab_bc__ac<T, S0, S1>(
                arg0: ndarray::ArrayBase<S0, ndarray::Ix2>,
                arg1: ndarray::ArrayBase<S1, ndarray::Ix2>,
            ) -> ndarray::Array<T, ndarray::Ix2>
            where
                T: ndarray::LinalgScalar,
                S0: ndarray::Data<Elem = T>,
                S1: ndarray::Data<Elem = T>,
            {
                let (n_a, n_b) = arg0.dim();
                let (_, n_c) = arg1.dim();
                {
                    let (n_0, n_1) = arg0.dim();
                    assert_eq!(n_0, n_a);
                    assert_eq!(n_1, n_b);
                }
                {
                    let (n_0, n_1) = arg1.dim();
                    assert_eq!(n_0, n_b);
                    assert_eq!(n_1, n_c);
                }
                let mut out0 = ndarray::Array::zeros((n_a, n_c));
                for a in 0..n_a {
                    for c in 0..n_c {
                        for b in 0..n_b {
                            out0[(a, c)] = arg0[(a, b)] * arg1[(b, c)];
                        }
                    }
                }
                out0
            }
            let arg0 = x;
            let arg1 = y;
            let out0 = ab_bc__ac(arg0, arg1);
            out0
        }
        "###);
    }

    #[test]
    fn einsum_ab_bc_cd() {
        let input = TokenStream2::from_str(r#""ab,bc,cd->ad", x, y, z"#).unwrap();
        let tt = format_block(einsum2(input).to_string());
        insta::assert_snapshot!(tt, @r###"
        {
            fn ab_bc__ac<T, S0, S1>(
                arg0: ndarray::ArrayBase<S0, ndarray::Ix2>,
                arg1: ndarray::ArrayBase<S1, ndarray::Ix2>,
            ) -> ndarray::Array<T, ndarray::Ix2>
            where
                T: ndarray::LinalgScalar,
                S0: ndarray::Data<Elem = T>,
                S1: ndarray::Data<Elem = T>,
            {
                let (n_a, n_b) = arg0.dim();
                let (_, n_c) = arg1.dim();
                {
                    let (n_0, n_1) = arg0.dim();
                    assert_eq!(n_0, n_a);
                    assert_eq!(n_1, n_b);
                }
                {
                    let (n_0, n_1) = arg1.dim();
                    assert_eq!(n_0, n_b);
                    assert_eq!(n_1, n_c);
                }
                let mut out1 = ndarray::Array::zeros((n_a, n_c));
                for a in 0..n_a {
                    for c in 0..n_c {
                        for b in 0..n_b {
                            out1[(a, c)] = arg0[(a, b)] * arg1[(b, c)];
                        }
                    }
                }
                out1
            }
            let arg0 = x;
            let arg1 = y;
            let arg2 = z;
            let out1 = ab_bc__ac(arg0, arg1);
            let out0 = ab_bc__ac(out1, arg2);
            out0
        }
        "###);
    }
}
