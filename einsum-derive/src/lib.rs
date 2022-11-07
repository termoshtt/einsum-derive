//! proc-macro based einsum implementation

use einsum_solver::subscripts::Subscripts;
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use proc_macro_error::{abort_call_site, proc_macro_error, OptionExt, ResultExt};
use quote::quote;
use std::{
    collections::{hash_map::Entry, HashMap},
    str::FromStr,
};
use syn::parse::Parser;

/// proc-macro based einsum
///
/// ```
/// use ndarray::array;
/// use einsum_derive::einsum;
///
/// let a = array![
///   [1.0, 2.0],
///   [3.0, 4.0]
/// ];
/// let b = array![
///   [1.0, 2.0],
///   [3.0, 4.0]
/// ];
/// let c = einsum!("ij,jk->ik", a, b);
/// assert_eq!(c, array![
///   [6.0, 8.0],
///   [12.0, 16.0]
/// ]);
/// ```
///
/// This proc-macro wil compile the input subscripts `"ij,jk->ik"`
/// to generate Rust code executing corresponding operation.
///
/// If the subscripts and the number of input mismatches,
/// this raises compile error:
///
/// ```compile_fail
/// use ndarray::array;
/// use einsum_derive::einsum;
///
/// let a = array![
///   [1.0, 2.0],
///   [3.0, 4.0]
/// ];
/// let c = einsum!("ij,jk->ik", a /* needs one more arg */);
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
        abort_call_site!(
            "Argument number mismatch: subscripts ({}), args ({})",
            subscripts.inputs.len(),
            args.len()
        );
    }

    let einsum_fn = def_einsum_fn(&subscripts);
    let fn_name = syn::Ident::new(&format!("{}", subscripts), Span::call_site());
    quote! {
        {
            #einsum_fn
            #fn_name(#(#args),*)
        }
    }
}

/// Generate for loop
///
/// ```ignore
/// for #index0 in 0..#n0 {
///     for #index1 in 0..#n1 {
///         for #index2 in 0..#n2 {
///            #inner
///         }
///     }
/// }
/// ```
fn contraction_for(indices: &[char], inner: TokenStream2) -> TokenStream2 {
    let mut tt = inner;
    for &i in indices.iter().rev() {
        let index = index_ident(i);
        let n = n_ident(i);
        tt = quote! {
            for #index in 0..#n { #tt }
        };
    }
    tt
}

fn contraction_inner(subscripts: &Subscripts) -> TokenStream2 {
    let mut inner_args_tt = Vec::new();
    for argc in 0..subscripts.inputs.len() {
        let name = arg_ident(argc);
        let mut index = Vec::new();
        for i in subscripts.inputs[argc].indices() {
            index.push(index_ident(i));
        }
        inner_args_tt.push(quote! {
            #name[(#(#index),*)]
        })
    }
    let mut inner_mul = None;
    for inner in inner_args_tt {
        match inner_mul {
            Some(i) => inner_mul = Some(quote! { #i * #inner }),
            None => inner_mul = Some(inner),
        }
    }

    let output_ident = output_ident();
    let mut output_indices = Vec::new();
    for i in &subscripts.output.indices() {
        let index = index_ident(*i);
        output_indices.push(index.clone());
    }
    quote! {
        #output_ident[(#(#output_indices),*)] = #inner_mul;
    }
}

/// Generate contraction parts, e.g.
///
/// ```ignore
/// for i in 0..n_i {
///     for k in 0..n_k {
///         for j in 0..n_j {
///             out[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
///         }
///     }
/// }
/// ```
///
fn contraction(subscripts: &Subscripts) -> TokenStream2 {
    let mut indices: Vec<char> = subscripts.output.indices();
    for i in subscripts.contraction_indices() {
        indices.push(i);
    }

    let inner = contraction_inner(subscripts);
    contraction_for(&indices, inner)
}

fn array_size(subscripts: &Subscripts) -> Vec<TokenStream2> {
    let mut n_idents: HashMap<char, proc_macro2::Ident> = HashMap::new();
    let mut tt = Vec::new();
    for argc in 0..subscripts.inputs.len() {
        let name = arg_ident(argc);
        let mut index = Vec::new();
        let mut n_index_each = Vec::new();
        let mut def_or_assert = Vec::new();
        for (m, i) in subscripts.inputs[argc].indices().into_iter().enumerate() {
            index.push(index_ident(i));
            let n = n_each_ident(argc, m);
            match n_idents.entry(i) {
                Entry::Occupied(entry) => {
                    let n_ = entry.get();
                    def_or_assert.push(quote! {
                        assert_eq!(#n_, #n);
                    });
                }
                Entry::Vacant(entry) => {
                    let n_ident = n_ident(i);
                    def_or_assert.push(quote! {
                        let #n_ident = #n;
                    });
                    entry.insert(n_ident);
                }
            }
            n_index_each.push(n);
        }
        tt.push(quote! {
            let (#(#n_index_each),*) = #name.dim();
            #( #def_or_assert )*
        });
    }
    tt
}

fn def_output_array(subscripts: &Subscripts) -> TokenStream2 {
    // Define output array
    let output_ident = output_ident();
    let mut n_output = Vec::new();
    for i in subscripts.output.indices() {
        n_output.push(n_ident(i));
    }
    quote! {
        let mut #output_ident = ndarray::Array::zeros((#(#n_output),*));
    }
}

fn def_einsum_fn(subscripts: &Subscripts) -> TokenStream2 {
    let fn_name = syn::Ident::new(&format!("{}", subscripts), Span::call_site());
    let n = subscripts.inputs.len();

    let args: Vec<syn::Ident> = (0..n).map(arg_ident).collect();
    let storages: Vec<syn::Ident> = (0..n).map(|n| quote::format_ident!("S{}", n)).collect();
    let dims: Vec<syn::Path> = subscripts
        .inputs
        .iter()
        .map(|ss| dim(ss.indices().len()))
        .collect();

    let out_dim = dim(subscripts.output.indices().len());

    let array_size = array_size(subscripts);
    let output_ident = output_ident();
    let output_tt = def_output_array(subscripts);
    let contraction_tt = contraction(subscripts);

    quote! {
        fn #fn_name<T, #(#storages),*>(
            #( #args: ndarray::ArrayBase<#storages, #dims> ),*
        ) -> ndarray::Array<T, #out_dim>
        where
            T: ndarray::LinalgScalar,
            #( #storages: ndarray::Data<Elem = T> ),*
        {
            #(#array_size)*
            #output_tt
            #contraction_tt
            #output_ident
        }
    }
}

fn dim(n: usize) -> syn::Path {
    let ix = quote::format_ident!("Ix{}", n);
    syn::parse_quote! { ndarray::#ix }
}

fn output_ident() -> syn::Ident {
    quote::format_ident!("out")
}

fn index_ident(i: char) -> syn::Ident {
    quote::format_ident!("{}", i)
}

fn n_ident(i: char) -> syn::Ident {
    quote::format_ident!("n_{}", i)
}

fn n_each_ident(argc: usize, i: usize) -> syn::Ident {
    quote::format_ident!("n_{}_{}", argc, i)
}

fn arg_ident(argc: usize) -> syn::Ident {
    quote::format_ident!("arg{}", argc)
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
            fn ij_jk__ik<T, S0, S1>(
                arg0: ndarray::ArrayBase<S0, ndarray::Ix2>,
                arg1: ndarray::ArrayBase<S1, ndarray::Ix2>,
            ) -> ndarray::Array<T, ndarray::Ix2>
            where
                T: ndarray::LinalgScalar,
                S0: ndarray::Data<Elem = T>,
                S1: ndarray::Data<Elem = T>,
            {
                let (n_0_0, n_0_1) = arg0.dim();
                let n_i = n_0_0;
                let n_j = n_0_1;
                let (n_1_0, n_1_1) = arg1.dim();
                assert_eq!(n_j, n_1_0);
                let n_k = n_1_1;
                let mut out = ndarray::Array::zeros((n_i, n_k));
                for i in 0..n_i {
                    for k in 0..n_k {
                        for j in 0..n_j {
                            out[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
                        }
                    }
                }
                out
            }
            ij_jk__ik(a, b)
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

        let formatted_lines: Vec<&str> = out
            .lines()
            .filter_map(|line| match line {
                "fn main() {" | "}" => None,
                _ => line.strip_prefix("    "),
            })
            .collect();
        formatted_lines.join("\n")
    }
}
