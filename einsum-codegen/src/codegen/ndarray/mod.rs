//! For [ndarray](https://crates.io/crates/ndarray) crate

mod naive;

pub use naive::naive;

use crate::subscripts::Subscripts;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashSet;

fn dim(n: usize) -> syn::Path {
    let ix = quote::format_ident!("Ix{}", n);
    syn::parse_quote! { ndarray::#ix }
}

pub fn n_ident(i: char) -> syn::Ident {
    quote::format_ident!("n_{}", i)
}

/// Define the index size identifiers, e.g. `n_i`
fn define_array_size(subscripts: &Subscripts) -> TokenStream2 {
    let mut appeared: HashSet<char> = HashSet::new();
    let mut tt = Vec::new();
    for arg in subscripts.inputs.iter() {
        let n_ident: Vec<syn::Ident> = arg
            .indices()
            .into_iter()
            .map(|i| {
                if appeared.contains(&i) {
                    quote::format_ident!("_")
                } else {
                    appeared.insert(i);
                    n_ident(i)
                }
            })
            .collect();
        tt.push(quote! {
            let (#(#n_ident),*) = #arg.dim();
        });
    }
    quote! { #(#tt)* }
}

/// Generate `assert_eq!` to check the size of user input tensors
fn array_size_asserts(subscripts: &Subscripts) -> TokenStream2 {
    let mut tt = Vec::new();
    for arg in &subscripts.inputs {
        // local variable, e.g. `n_2`
        let n_each: Vec<_> = (0..arg.indices().len())
            .map(|m| quote::format_ident!("n_{}", m))
            .collect();
        // size of index defined previously, e.g. `n_i`
        let n: Vec<_> = arg.indices().into_iter().map(n_ident).collect();
        tt.push(quote! {
            let (#(#n_each),*) = #arg.dim();
            #(assert_eq!(#n_each, #n);)*
        });
    }
    quote! { #({ #tt })* }
}

/// Generate einsum function definition
pub fn function_definition(subscripts: &Subscripts, inner: TokenStream2) -> TokenStream2 {
    let fn_name = format_ident!("{}", subscripts.escaped_ident());
    let n = subscripts.inputs.len();

    let args = &subscripts.inputs;
    let storages: Vec<syn::Ident> = (0..n).map(|n| quote::format_ident!("S{}", n)).collect();
    let dims: Vec<syn::Path> = subscripts
        .inputs
        .iter()
        .map(|ss| dim(ss.indices().len()))
        .collect();

    let out_dim = dim(subscripts.output.indices().len());

    let array_size = define_array_size(subscripts);
    let array_size_asserts = array_size_asserts(subscripts);

    quote! {
        fn #fn_name<T, #(#storages),*>(
            #( #args: ndarray::ArrayBase<#storages, #dims> ),*
        ) -> ndarray::Array<T, #out_dim>
        where
            T: ndarray::LinalgScalar,
            #( #storages: ndarray::Data<Elem = T> ),*
        {
            #array_size
            #array_size_asserts

            #inner
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{codegen::format_block, *};

    #[test]
    fn define_array_size() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::define_array_size(&subscripts).to_string());
        insta::assert_snapshot!(tt, @r###"
        let (n_a, n_b) = arg0.dim();
        let (_, n_c) = arg1.dim();
        "###);
    }

    #[test]
    fn function_definition_snapshot() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let inner = quote::quote! { todo!() };
        let tt = format_block(super::function_definition(&subscripts, inner).to_string());
        insta::assert_snapshot!(tt, @r###"
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
            todo!()
        }
        "###);
    }
}
