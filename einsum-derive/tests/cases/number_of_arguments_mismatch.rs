use einsum_derive::einsum;
use ndarray::array;

fn main() {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let c = einsum!("ij,jk->ik", a /* needs one more arg */);
}
