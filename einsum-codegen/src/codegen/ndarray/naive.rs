//! Generate einsum function with naive loop

#[cfg(doc)]
use super::function_definition;

use crate::Subscripts;

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashSet;

fn index_ident(i: char) -> syn::Ident {
    quote::format_ident!("{}", i)
}

fn n_ident(i: char) -> syn::Ident {
    quote::format_ident!("n_{}", i)
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
        let n: Vec<_> = arg.indices().into_iter().map(n_ident).collect();
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

/// Actual component of einsum [function_definition]
pub fn inner(subscripts: &Subscripts) -> TokenStream2 {
    let array_size = define_array_size(subscripts);
    let array_size_asserts = array_size_asserts(subscripts);
    let output_ident = &subscripts.output;
    let output_tt = define_output_array(subscripts);
    let contraction_tt = contraction(subscripts);
    quote! {
        #array_size
        #array_size_asserts
        #output_tt
        #contraction_tt
        #output_ident
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
    fn contraction() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::contraction(&subscripts).to_string());
        insta::assert_snapshot!(tt, @r###"
        for a in 0..n_a {
            for c in 0..n_c {
                for b in 0..n_b {
                    out0[(a, c)] = arg0[(a, b)] * arg1[(b, c)];
                }
            }
        }
        "###);
    }

    #[test]
    fn inner() {
        let mut namespace = Namespace::init();
        let subscripts = Subscripts::from_raw_indices(&mut namespace, "ij,jk->ik").unwrap();
        let tt = format_block(super::inner(&subscripts).to_string());
        insta::assert_snapshot!(tt, @r###"
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
