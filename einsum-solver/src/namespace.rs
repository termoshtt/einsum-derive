/// Names of tensors
///
/// As the crate level document explains,
/// einsum factorization requires to track names of tensors
/// in addition to subscripts, and this struct manages it.
/// This works as a simple counter, which counts how many intermediate
/// tensor denoted `out{N}` appears and issues new `out{N+1}` identifier.
///
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Namespace {
    last: usize,
}

impl Namespace {
    /// Create new namespace
    pub fn init() -> Self {
        Namespace { last: 0 }
    }

    /// Issue new identifier
    pub fn new_ident(&mut self) -> Position {
        let pos = Position::Intermidiate(self.last);
        self.last += 1;
        pos
    }
}

/// Which tensor the subscript specifies
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum Position {
    /// The tensor which user inputs as N-th argument of einsum
    User(usize),
    /// The tensor created by einsum in its N-th step
    Intermidiate(usize),
}
