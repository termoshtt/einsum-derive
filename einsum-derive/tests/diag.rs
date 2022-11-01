use einsum_derive::einsum;
use ndarray::array;

#[test]
fn diag() {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let d = einsum!("ii->i", a);
    assert_eq!(d, array![1.0, 4.0]);
}
