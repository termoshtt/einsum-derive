//! proc-macro based einsum implementation

use einsum_solver::subscripts::Subscripts;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::{abort_call_site, proc_macro_error, OptionExt, ResultExt};
use quote::quote;
use std::str::FromStr;
use syn::parse::Parser;

/// proc-macro based einsum
///
/// ```
/// let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
/// let b = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
/// let c = einsum_derive::einsum!("ij,jk->ik", a, b);
/// ```
///
/// Number of input mismatch causes compile error:
///
/// ```compile_fail
/// let a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
/// let c = einsum_derive::einsum!("ij,jk->ik", a);
/// ```
#[proc_macro_error]
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    einsum2(input.into()).into()
}

fn einsum2(input: TokenStream2) -> TokenStream2 {
    let (subscripts, args) = parse_einsum_args(input);

    // Validate subscripts
    let subscripts = Subscripts::from_str(&subscripts)
        .ok()
        .expect_or_abort("Invalid subscripts");
    if subscripts.inputs.len() != args.len() {
        abort_call_site!("Argument number mismatch");
    }

    let names: Vec<syn::Ident> = (0..args.len())
        .map(|i| quote::format_ident!("arg{}", i))
        .collect();

    // Generate a block which returns result tensor
    quote! {
        {
            #( let #names = #args; )*
            ()
        }
    }
}

fn parse_einsum_args(input: TokenStream2) -> (String, Vec<syn::Expr>) {
    let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
    let args = parser
        .parse2(input)
        .expect_or_abort("Invalid input for einsum!");
    let mut iter = args.into_iter();
    let subscripts = if let Some(syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit),
        attrs: _,
    })) = iter.next()
    {
        lit.value()
    } else {
        abort_call_site!("einsum! must start with subscript string literal")
    };
    let args = iter.collect::<Vec<_>>();
    (subscripts, args)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_snapshots() {
        let input = TokenStream2::from_str(r#""ij,jk->ik", a, b"#).unwrap();
        let tt = einsum2(input).to_string();
        insta::assert_snapshot!(tt, @"{ let arg0 = a ; let arg1 = b ; () }");
    }
}
