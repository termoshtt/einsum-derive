use proc_macro2::TokenStream as TokenStream2;
use syn::parse::Parser;

pub fn parse(input: TokenStream2) -> (String, Vec<syn::Expr>) {
    let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
    let args = parser.parse2(input).expect("Invalid input for einsum!");
    let mut iter = args.into_iter();
    let subscripts = if let Some(syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Str(lit),
        attrs: _,
    })) = iter.next()
    {
        lit.value()
    } else {
        panic!("einsum! must start with subscript string literal")
    };
    let args = iter.collect::<Vec<_>>();
    (subscripts, args)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_parse() {
        let input = TokenStream2::from_str(r#""ij,jk->ik", a, b"#).unwrap();
        let (subscripts, exprs) = parse(input);
        assert_eq!(subscripts, "ij,jk->ik");
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0], syn::parse_str::<syn::Expr>("a").unwrap());
        assert_eq!(exprs[1], syn::parse_str::<syn::Expr>("b").unwrap());
    }
}
