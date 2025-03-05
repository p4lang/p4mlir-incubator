// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK:   p4hir.func @externFunc<!p4hir.type_var<"T">, !p4hir.type_var<"U">>(!p4hir.type_var<"U"> {p4hir.dir = #in}, !p4hir.type_var<"T"> {p4hir.dir = #in}) -> !p4hir.type_var<"T">
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
