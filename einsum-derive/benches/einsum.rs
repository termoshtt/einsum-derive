use criterion::*;
use einsum_derive::einsum;
use ndarray::*;
use ndarray_linalg::*;

fn einsum_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("ij_jk", n), &n, |bench, n| {
            let a: Array2<f64> = random((*n, *n));
            let b: Array2<f64> = random((*n, *n));
            bench.iter(|| {
                let _c = einsum!("ij,jk", a.clone(), b.clone());
            })
        });

        group.bench_with_input(BenchmarkId::new("ij_jk_kl", n), &n, |bench, n| {
            let a: Array2<f64> = random((*n, *n));
            let b: Array2<f64> = random((*n, *n));
            let c: Array2<f64> = random((*n, *n));
            bench.iter(|| {
                let _c = einsum!("ij,jk,kl", a.clone(), b.clone(), c.clone());
            })
        });
    }
}

criterion_group!(einsum, einsum_bench);
criterion_main!(einsum);
