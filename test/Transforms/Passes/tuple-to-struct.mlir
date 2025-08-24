// RUN: p4mlir-opt --p4hir-tuple-to-struct %s | FileCheck %s

!b32i = !p4hir.bit<32>
!i32i = !p4hir.int<32>

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @basic

// CHECK: %[[S:.*]] = p4hir.struct %{{.*}}, %{{.*}}, : i32, i1
// CHECK: %[[X:.*]] = p4hir.struct_extract %[[S]][0] : ! !p4hir.struct<i32, i1> -> i32
// CHECK: return %[[X]] : i32
module {

   p4hir.func @basic(%x: !b32i, %y: !i32i) -> tuple<!b32i, !i32i> {
        %t = p4hir.tuple(%x, %y) : tuple<!b32i, !i32i>
        //%x = p4hir.tuple_extract %t[0] : !p4hir.tuple<i32, i1> -> i32
        p4hir.return %t : tuple<!b32i, !i32i>
    }


}