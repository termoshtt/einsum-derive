//! proc-macro based einsum implementation

use einsum_solver::subscripts::Subscripts;
use proc_macro::{TokenStream, TokenTree};
use proc_macro_error::*;
use quote::quote;
use std::str::FromStr;

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
    // Check proc-macro input
    let mut iter = input.into_iter();
    let subscripts = if let Some(TokenTree::Literal(lit)) = iter.next() {
        lit.to_string().trim_matches('"').to_string()
    } else {
        abort_call_site!("einsum! must start with subscript string literal");
    };
    let mut args = Vec::new();
    while let Some(arg) = iter.next() {
        match arg {
            TokenTree::Ident(ident) => args.push(ident),
            _ => continue,
        }
    }

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
