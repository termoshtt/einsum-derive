use einsum_derive::einsum;
use ndarray::array;

#[test]
fn matmul() {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[1.0, 2.0], [3.0, 4.0]];
    let c = einsum!("ij,jk->ik", a, b);
    assert_eq!(c, array![[6.0, 8.0], [12.0, 16.0]]);
}
