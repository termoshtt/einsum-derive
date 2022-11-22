pub fn dim(n: usize) -> syn::Path {
    let ix = quote::format_ident!("Ix{}", n);
    syn::parse_quote! { ndarray::#ix }
}

pub fn output_ident() -> syn::Ident {
    quote::format_ident!("out")
}

pub fn index_ident(i: char) -> syn::Ident {
    quote::format_ident!("{}", i)
}

pub fn n_ident(i: char) -> syn::Ident {
    quote::format_ident!("n_{}", i)
}

pub fn n_each_ident(argc: usize, i: usize) -> syn::Ident {
    quote::format_ident!("n_{}_{}", argc, i)
}

pub fn arg_ident(argc: usize) -> syn::Ident {
    quote::format_ident!("arg{}", argc)
}
