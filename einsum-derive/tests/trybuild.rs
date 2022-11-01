#[test]
fn trybuild() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/cases/number_of_arguments_mismatch.rs");
}
