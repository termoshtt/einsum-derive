use einsum_solver::subscripts::Subscripts;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::{hash_map::Entry, HashMap};

use crate::ident::*;

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

pub fn def_einsum_fn(subscripts: &Subscripts) -> TokenStream2 {
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
