//! proc-macro based einsum implementation

use proc_macro::{TokenStream, TokenTree};
use proc_macro_error::*;

/// proc-macro based einsum
///
/// ```
/// use einsum_derive::einsum;
///
/// let c = einsum!("ij,jk->ik", a, b);
/// ```
#[proc_macro_error]
#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let subscripts = if let Some(TokenTree::Literal(lit)) = iter.next() {
        lit.to_string()
    } else {
        abort_call_site!("einsum! must start with subscript string literal");
    };

    dbg!(subscripts);

    todo!()
}
