// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK:  ![[type_T:.*]] = !p4hir.type_var<"T">
// CHECK:  ![[type_U:.*]] = !p4hir.type_var<"U">

// CHECK:   p4hir.func @externFunc<![[type_T]], ![[type_U]]>(![[type_U]] {p4hir.dir = #in, p4hir.param_name = "a"}, ![[type_T]] {p4hir.dir = #in, p4hir.param_name = "b"}) -> ![[type_T]]
extern T externFunc<T, U>(in U a, in T b);

parser p1() {
    state start {
      // CHECK: p4hir.call @externFunc<[!b8i, !b16i]> (%{{.*}}, %{{.*}}) : (!b16i, !b8i) -> !b8i
      bit<8> res = externFunc(16w0, 8w0);
      transition accept;
    }
}

action foo(inout bit<32> a, in int<7> b) {
    // CHECK: p4hir.call @externFunc<[!b32i, !i7i]> (%{{.*}}, %{{.*}}) : (!i7i, !b32i) -> !b32i
    a = externFunc(b, a);
}
