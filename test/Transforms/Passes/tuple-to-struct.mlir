// RUN: p4mlir-opt --p4hir-tuple-to-struct %s | FileCheck %s

!b32i = !p4hir.bit<32>
!i32i = !p4hir.int<32>

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @bid
// CHECK: p4hir.return
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @bid(%t : tuple<!b32i, !i32i>) ->  tuple<!b32i, !i32i>{
        p4hir.return %t :  tuple<!b32i, !i32i>
    }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @make_tuple
// CHECK: p4hir.return
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @make_tuple(%x : !b32i, %y: !i32i) -> tuple<!b32i, !i32i> {
        %t = p4hir.tuple(%x, %y) : tuple<!b32i, !i32i>
        p4hir.return %t : tuple<!b32i, !i32i>
    }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @make_large_tuple
// CHECK: p4hir.return
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @make_large_tuple(%x : !b32i, %y: !i32i, %z: !b32i) -> tuple<!b32i, !i32i, !b32i> {
        %t = p4hir.tuple(%x, %y, %z) : tuple<!b32i, !i32i, !b32i>
        p4hir.return %t : tuple<!b32i, !i32i, !b32i>
    }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @create_and_return
// CHECK: p4hir.return
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @create_and_return(%x : !b32i, %y: !i32i) -> tuple<!b32i, !i32i> {
        %t = p4hir.tuple(%x, %y) : tuple<!b32i, !i32i>
        p4hir.return %t : tuple<!b32i, !i32i>
    }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @create_nested
// CHECK: p4hir.return
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @create_nested(%x : !b32i, %y: !i32i, %z: !b32i) -> tuple<tuple<!b32i, !i32i>, !b32i> {
        %inner = p4hir.tuple(%x, %y) : tuple<!b32i, !i32i>
        %outer = p4hir.tuple(%inner, %z) : tuple<tuple<!b32i, !i32i>, !b32i>
        p4hir.return %outer : tuple<tuple<!b32i, !i32i>, !b32i>
    }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @nested_extract
// CHECK : p4hir.struct_extract
// CHECK: p4hir.return
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @nested_extract(%t : tuple<tuple<!b32i, !i32i>, !b32i>) -> !i32i {
        %inner = p4hir.tuple_extract %t[0] : tuple<tuple<!b32i, !i32i>, !b32i>
        %e = p4hir.tuple_extract %inner[1] : tuple<!b32i, !i32i>
        p4hir.return %e : !i32i
    }
}