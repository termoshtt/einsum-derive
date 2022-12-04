//! Generate einsum function with naive loop

use super::*;
use crate::Subscripts;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

fn index_ident(i: char) -> syn::Ident {
    quote::format_ident!("{}", i)
}

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

/// Generate naive contraction loop
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
fn contraction(subscripts: &Subscripts) -> TokenStream2 {
    let mut indices: Vec<char> = subscripts.output.indices();
    for i in subscripts.contraction_indices() {
        indices.push(i);
    }

    let inner = contraction_inner(subscripts);
    contraction_for(&indices, inner)
}

/// Actual component of einsum [function_definition]
pub fn naive(subscripts: &Subscripts) -> TokenStream2 {
    let output_ident = &subscripts.output;
    let contraction_tt = contraction(subscripts);
    let n_output: Vec<_> = subscripts
        .output
        .indices()
        .into_iter()
        .map(n_ident)
        .collect();
    quote! {
        let mut #output_ident = ndarray::Array::zeros((#(#n_output),*));
        #contraction_tt
        #output_ident
    }
}

#[cfg(test)]
mod test {
    use crate::{codegen::format_block, *};

    #[test]
    fn naive() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::naive(&subscripts).to_string());
        insta::assert_snapshot!(tt, @r###"
        let mut out0 = ndarray::Array::zeros((n_a, n_c));
        for a in 0..n_a {
            for c in 0..n_c {
                for b in 0..n_b {
                    out0[(a, c)] = arg0[(a, b)] * arg1[(b, c)];
                }
            }
        }
        out0
        "###);
    }
}
