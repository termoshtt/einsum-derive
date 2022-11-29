//! For [ndarray](https://crates.io/crates/ndarray) crate

pub mod naive;

use crate::{namespace::Position, subscripts::Subscripts};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;

fn dim(n: usize) -> syn::Path {
    let ix = quote::format_ident!("Ix{}", n);
    syn::parse_quote! { ndarray::#ix }
}

pub fn function_definition(subscripts: &Subscripts, inner: TokenStream2) -> TokenStream2 {
    let fn_name = syn::Ident::new(&subscripts.escaped_ident(), Span::call_site());
    let n = subscripts.inputs.len();

    let args: Vec<_> = (0..n).map(|n| Position::Arg(n)).collect();
    let storages: Vec<syn::Ident> = (0..n).map(|n| quote::format_ident!("S{}", n)).collect();
    let dims: Vec<syn::Path> = subscripts
        .inputs
        .iter()
        .map(|ss| dim(ss.indices().len()))
        .collect();

    let out_dim = dim(subscripts.output.indices().len());

    quote! {
        fn #fn_name<T, #(#storages),*>(
            #( #args: ndarray::ArrayBase<#storages, #dims> ),*
        ) -> ndarray::Array<T, #out_dim>
        where
            T: ndarray::LinalgScalar,
            #( #storages: ndarray::Data<Elem = T> ),*
        {
            #inner
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{codegen::format_block, namespace::Namespace, subscripts::Subscripts};

    #[test]
    fn function_definition_snapshot() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let inner = quote::quote! { todo!() };
        let tt = format_block(super::function_definition(&subscripts, inner).to_string());
        insta::assert_snapshot!(tt, @r###"
        fn ij_jk__ik<T, S0, S1>(
            arg0: ndarray::ArrayBase<S0, ndarray::Ix2>,
            arg1: ndarray::ArrayBase<S1, ndarray::Ix2>,
        ) -> ndarray::Array<T, ndarray::Ix2>
        where
            T: ndarray::LinalgScalar,
            S0: ndarray::Data<Elem = T>,
            S1: ndarray::Data<Elem = T>,
        {
            todo!()
        }
        "###);
    }
}
