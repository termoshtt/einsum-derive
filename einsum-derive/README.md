einsum-derive
===============
Proc-macro based einsum implementation for [ndarray](https://crates.io/crates/ndarray) crate

```rust
use ndarray::array;
use einsum_derive::einsum;

let a = array![
  [1.0, 2.0],
  [3.0, 4.0]
];
let b = array![
  [1.0, 2.0],
  [3.0, 4.0]
];
let c = einsum!("ij,jk->ik", a, b);
assert_eq!(c, array![
  [6.0, 8.0],
  [12.0, 16.0]
]);
```

This proc-macro wil compile the input subscripts `"ij,jk->ik"`
to generate Rust code executing corresponding operation.

Status / Roadmap
-----------------
- [x] [Optimal contraction by memorizing partial summation to reduce computation order.](https://github.com/termoshtt/einsum-derive/pull/18)
  - For example, three matrix multiplication `ij,jk,kl->il` is factorized into
    two successive einsum `ij,jk->ik` and `ik,kl->il`.
- [ ] [Call BLAS routines if possible](https://github.com/termoshtt/einsum-derive/issues/22)
- [ ] [Ellipsis `...` support](https://github.com/termoshtt/einsum-derive/issues/7)

Architecture
-------------
| crates.io | docs.rs | GitHub Pages | Description |
|:---------:|:-------:|:------------:|:------------|
| [![crate](https://img.shields.io/crates/v/einsum-derive.svg)](https://crates.io/crates/einsum-derive) | [![docs.rs](https://docs.rs/einsum-derive/badge.svg)](https://docs.rs/einsum-derive) | [![Pages](https://img.shields.io/badge/docs-main-blue)](https://termoshtt.github.io/einsum-derive/doc/einsum_derive/index.html) | A crate is proc-macro crate to provide above `einsum!` macro. |
| [![crate](https://img.shields.io/crates/v/einsum-codegen.svg)](https://crates.io/crates/einsum-codegen) | [![docs.rs](https://docs.rs/einsum-codegen/badge.svg)](https://docs.rs/einsum-codegen) | [![Pages](https://img.shields.io/badge/docs-main-blue)](https://termoshtt.github.io/einsum-codegen/doc/einsum_codegen/index.html) | A crate implements parser for the einsum subscripts like `ij,jk->ik` and generates Rust code. |

Benchmark
----------
[![bench](https://img.shields.io/badge/benchmark-main-orange)](https://termoshtt.github.io/einsum-derive/bench/report/index.html)

Benchmark with [criterion.rs](https://github.com/bheisler/criterion.rs) is running on GitHub Action on every commit on the main branch.

Links
------
- [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) is well-known einsum implementation in Python.
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) is an implementation for optimizing einsum computation for NumPy and other linear algebra packages.
- [oracleofnj/einsum](https://github.com/oracleofnj/einsum) is a runtime-based implementation of einsum for rust-ndarray

License
--------

© 2022 Toshiki Teramura (@termoshtt)

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
