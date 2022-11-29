use criterion::*;
use einsum_derive::einsum;
use ndarray::*;
use ndarray_linalg::*;

fn ij_jk(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("ij_jk", n), &n, |bench, n| {
            let a: Array2<f64> = random((*n, *n));
            let b: Array2<f64> = random((*n, *n));
            bench.iter(|| {
                let _c = einsum!("ij,jk", a.clone(), b.clone());
            })
        });
    }
}

criterion_group!(einsum, ij_jk);
criterion_main!(einsum);
