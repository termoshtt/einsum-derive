use std::fmt;

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
        let pos = Position::Out(self.last);
        self.last += 1;
        pos
    }
}

/// Which tensor the subscript specifies
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum Position {
    /// The tensor which user inputs as N-th argument of einsum
    Arg(usize),
    /// The tensor created by einsum in its N-th step
    Out(usize),
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Position::Arg(n) => write!(f, "arg{}", n),
            Position::Out(n) => write!(f, "out{}", n),
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
