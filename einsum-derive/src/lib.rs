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
/// use einsum_derive::einsum;
///
/// let c = einsum!("ij,jk->ik", a, b);
/// ```
///
/// ```compile_fail
/// use einsum_derive::einsum;
///
/// // Number of input mismatches!
/// let c = einsum!("ij,jk->ik", a);
/// ```
#[proc_macro_error]
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    let (subscripts, args) = parse_einsum_args(input.into());

    // Validate subscripts
    let subscripts = Subscripts::from_str(&subscripts)
        .ok()
        .expect_or_abort("Invalid subscripts");
    if subscripts.inputs.len() != args.len() {
        abort_call_site!("Argument number mismatch");
    }

    // Generate summation
    quote! {
        ()
    }
    .into()
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
