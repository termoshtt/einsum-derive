einsum-derive
===============
[![master](https://img.shields.io/badge/docs-master-blue)](https://termoshtt.github.io/einsum-derive/doc/einsum_derive/index.html)
[![bench](https://img.shields.io/badge/benchmark-master-orange)](https://termoshtt.github.io/einsum-derive/bench/report/index.html)

Proc-macro based einsum implementation for rust-ndarray

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

Examples
---------

- `matmul3`

  ```rust
  use ndarray::array;
  use einsum_derive::einsum;

  let a = array![[1.0, 2.0], [3.0, 4.0]];
  let b = array![[1.0, 2.0], [3.0, 4.0]];
  let c = array![[1.0, 2.0], [3.0, 4.0]];
  let d = einsum!("ij,jk,kl->il", a, b, c);
  assert_eq!(d, array![[24.0, 32.0], [48.0, 64.0]]);
  ```

- Take diagonal elements

  ```rust
  use ndarray::array;
  use einsum_derive::einsum;

  let a = array![[1.0, 2.0], [3.0, 4.0]];
  let d = einsum!("ii->i", a);
  assert_eq!(d, array![1.0, 4.0]);
  ```

- If the subscripts and the number of input mismatches,
  this raises compile error:

  ```compile_fail
  use ndarray::array;
  use einsum_derive::einsum;

  let a = array![
    [1.0, 2.0],
    [3.0, 4.0]
  ];
  let c = einsum!("ij,jk->ik", a /* needs one more arg */);
  ```

Links
------
- [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) is well-known einsum implementation.
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) is an implementation for optimizing einsum computation.
- [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) is a Rust implementation of ndarray.
  - [Einsum support: From external crate #960](https://github.com/rust-ndarray/ndarray/issues/960)
- [oracleofnj/einsum](https://github.com/oracleofnj/einsum) is a runtime-based implementation of einsum for rust-ndarray
