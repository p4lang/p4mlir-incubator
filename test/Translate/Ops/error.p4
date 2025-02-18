// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK: !error = !p4hir.error<Foo, Bar, Baz>
// CHECK: !S = !p4hir.struct<"S", e: !error>
// CHECK: #error_Baz = #p4hir.error<Baz, !error> : !error
// CHECK: #error_Foo = #p4hir.error<Foo, !error> : !error

error { Foo, Bar };

struct S {
 error e;
};

action test(inout S s) {
// CHECK-LABEL: test
// CHECK: %[[e_field_ref:.*]] = p4hir.struct_extract_ref %arg0["e"] : <!S>
// CHECK: %[[error_Foo:.*]] = p4hir.const #error_Foo
// CHECK: p4hir.assign %[[error_Foo]], %[[e_field_ref]] : <!error>
  s.e = error.Foo;
}

error { Baz }

const S s = { error.Baz };
