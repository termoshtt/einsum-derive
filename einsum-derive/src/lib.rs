//! proc-macro based einsum implementation

use proc_macro::TokenStream;
use proc_macro_error::proc_macro_error;

/// proc-macro based einsum
///
/// ```
/// use einsum_derive::einsum;
///
/// let c = einsum!("ij,jk->ik", a, b);
/// ```
#[proc_macro_error]
#[proc_macro]
pub fn einsum(_input: TokenStream) -> TokenStream {
    todo!()
}
