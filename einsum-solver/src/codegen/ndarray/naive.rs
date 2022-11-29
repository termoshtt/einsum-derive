//! Generate einsum function with naive loop

use crate::{namespace::Position, subscripts::Subscripts};

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::HashSet;

fn dim(n: usize) -> syn::Path {
    let ix = quote::format_ident!("Ix{}", n);
    syn::parse_quote! { ndarray::#ix }
}

fn index_ident(i: char) -> syn::Ident {
    quote::format_ident!("{}", i)
}

fn n_ident(i: char) -> syn::Ident {
    quote::format_ident!("n_{}", i)
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
    for (argc, arg) in subscripts.inputs.iter().enumerate() {
        let mut index = Vec::new();
        for i in subscripts.inputs[argc].indices() {
            index.push(index_ident(i));
        }
        inner_args_tt.push(quote! {
            #arg[(#(#index),*)]
        })
    }
    let mut inner_mul = None;
    for inner in inner_args_tt {
        match inner_mul {
            Some(i) => inner_mul = Some(quote! { #i * #inner }),
            None => inner_mul = Some(inner),
        }
    }

    let output_ident = &subscripts.output;
    let mut output_indices = Vec::new();
    for i in &subscripts.output.indices() {
        let index = index_ident(*i);
        output_indices.push(index.clone());
    }
    quote! {
        #output_ident[(#(#output_indices),*)] = #inner_mul;
    }
}

/// Generate naive contraction loop, e.g.
///
/// ```
/// # use ndarray::Array2;
/// # let arg0 = Array2::<f64>::zeros((3, 3));
/// # let arg1 = Array2::<f64>::zeros((3, 3));
/// # let mut out0 = Array2::<f64>::zeros((3, 3));
/// # let n_i = 3;
/// # let n_j = 3;
/// # let n_k = 3;
/// for i in 0..n_i {
///     for k in 0..n_k {
///         for j in 0..n_j {
///             out0[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
///         }
///     }
/// }
/// ```
///
pub fn contraction(subscripts: &Subscripts) -> TokenStream2 {
    let mut indices: Vec<char> = subscripts.output.indices();
    for i in subscripts.contraction_indices() {
        indices.push(i);
    }

    let inner = contraction_inner(subscripts);
    contraction_for(&indices, inner)
}

/// Define the index size identifiers, e.g. `n_i`
pub fn define_array_size(subscripts: &Subscripts) -> TokenStream2 {
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
pub fn array_size_asserts(subscripts: &Subscripts) -> TokenStream2 {
    let mut tt = Vec::new();
    for arg in &subscripts.inputs {
        // local variable, e.g. `n_2`
        let n_each: Vec<_> = (0..arg.indices().len())
            .map(|m| quote::format_ident!("n_{}", m))
            .collect();
        // size of index defined previously, e.g. `n_i`
        let n: Vec<_> = arg.indices().into_iter().map(|i| n_ident(i)).collect();
        tt.push(quote! {
            let (#(#n_each),*) = #arg.dim();
            #(assert_eq!(#n_each, #n);)*
        });
    }
    quote! { #({ #tt })* }
}

fn define_output_array(subscripts: &Subscripts) -> TokenStream2 {
    // Define output array
    let output_ident = &subscripts.output;
    let mut n_output = Vec::new();
    for i in subscripts.output.indices() {
        n_output.push(n_ident(i));
    }
    quote! {
        let mut #output_ident = ndarray::Array::zeros((#(#n_output),*));
    }
}

pub fn define(subscripts: &Subscripts) -> TokenStream2 {
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

    let array_size = define_array_size(subscripts);
    let array_size_asserts = array_size_asserts(subscripts);
    let output_ident = &subscripts.output;
    let output_tt = define_output_array(subscripts);
    let contraction_tt = contraction(subscripts);

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
            #output_tt
            #contraction_tt
            #output_ident
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{codegen::format_block, namespace::Namespace, subscripts::Subscripts};

    #[test]
    fn define_array_size() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::define_array_size(&subscripts).to_string());
        insta::assert_snapshot!(tt, @r###"
        let (n_i, n_j) = arg0.dim();
        let (_, n_k) = arg1.dim();
        "###);
    }

    #[test]
    fn contraction_snapshots() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::contraction(&subscripts).to_string());
        insta::assert_snapshot!(tt, @r###"
        for i in 0..n_i {
            for k in 0..n_k {
                for j in 0..n_j {
                    out0[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
                }
            }
        }
        "###);
    }

    #[test]
    fn einsum_fn_snapshots() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::define(&subscripts).to_string());
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
            let (n_i, n_j) = arg0.dim();
            let (_, n_k) = arg1.dim();
            {
                let (n_0, n_1) = arg0.dim();
                assert_eq!(n_0, n_i);
                assert_eq!(n_1, n_j);
            }
            {
                let (n_0, n_1) = arg1.dim();
                assert_eq!(n_0, n_j);
                assert_eq!(n_1, n_k);
            }
            let mut out0 = ndarray::Array::zeros((n_i, n_k));
            for i in 0..n_i {
                for k in 0..n_k {
                    for j in 0..n_j {
                        out0[(i, k)] = arg0[(i, j)] * arg1[(j, k)];
                    }
                }
            }
            out0
        }
        "###);
    }
}
