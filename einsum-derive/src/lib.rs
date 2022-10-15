//! proc-macro based einsum implementation

use einsum_solver::subscripts::{Label, Subscripts};
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

    let mut tt = Vec::new();
    for argc in 0..args.len() {
        let name = quote::format_ident!("arg{}", argc);
        let arg = &args[argc];
        let ns: Vec<_> = subscripts.inputs[argc]
            .iter()
            .map(|label| match label {
                Label::Index(i) => quote::format_ident!("n_{}_{}", argc, i),
                _ => unimplemented!(),
            })
            .collect();
        tt.push(quote! {
            let #name = #arg;
            let (#(#ns),*) = #name.dim();
        });
    }

    // Generate a block which returns result tensor
    quote! {
        {
            #(#tt)*
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
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    #[test]
    fn test_snapshots() {
        let input = TokenStream2::from_str(r#""ij,jk->ik", a, b"#).unwrap();
        let tt = format_block(einsum2(input).to_string());
        insta::assert_snapshot!(tt, @r###"
            {
                let arg0 = a;
                let (n_0_i, n_0_j) = arg0.dim();
                let arg1 = b;
                let (n_1_j, n_1_k) = arg1.dim();
                ()
            }
        "###);
    }

    /// Format generated Rust code using `rustfmt` run as external process.
    pub fn format_block(tt: String) -> String {
        let tt = format!("fn main() {{ {} }}", tt);

        let mut child = Command::new("rustfmt")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to spawn rustfmt process");

        // Write input from another thread for avoiding deadlock.
        // See https://doc.rust-lang.org/std/process/index.html#handling-io
        let mut stdin = child.stdin.take().expect("Failed to open stdin");
        std::thread::spawn(move || {
            stdin
                .write_all(tt.as_bytes())
                .expect("Failed to write to stdin");
        });
        let output = child
            .wait_with_output()
            .expect("Failed to wait output of rustfmt process");

        // non-UTF8 comment should be handled in the tokenize phase,
        // and not be included in IR.
        let out = String::from_utf8(output.stdout).expect("rustfmt output contains non-UTF8 input");

        out.strip_prefix("fn main() {\n")
            .and_then(|out| out.strip_suffix("}\n"))
            .unwrap()
            .to_string()
    }
}
