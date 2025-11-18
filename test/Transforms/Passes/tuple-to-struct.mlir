// RUN: p4mlir-opt --split-input-file --p4hir-tuple-to-struct %s | FileCheck %s

!b32i = !p4hir.bit<32>
!i32i = !p4hir.int<32>
!_tupleToStruct = !p4hir.struct<"_tupleToStruct", element_0: !b32i, element_1: !i32i>

// CHECK: ![[tupleToStructType:_tupleToStruct]] = !p4hir.struct<"_tupleToStruct", element_0: !b32i, element_1: !i32i>
// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @bid
// CHECK-SAME: (%arg0: ![[tupleToStructType]]) -> ![[tupleToStructType]]
// CHECK: p4hir.return %arg0 : ![[tupleToStructType]]
// CHECK-NOT: tuple<
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
  p4hir.func @bid(%t : tuple<!b32i, !i32i>) -> tuple<!b32i, !i32i> {
    p4hir.return %t : tuple<!b32i, !i32i>
  }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @make_tuple
// CHECK-SAME: (%arg0: !b32i, %arg1: !i32i) -> !_tupleToStruct
// CHECK: %[[S:.*]]  = p4hir.struct (%arg0, %arg1) : !_tupleToStruct
// CHECK: p4hir.return %[[S]] : !_tupleToStruct
// CHECK-NOT: tuple<
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
// CHECK-SAME: (%arg0: !b32i, %arg1: !i32i, %arg2: !b32i)
// CHECK: %[[STRUCT:.*]] = p4hir.struct (%arg0, %arg1, %arg2)
// CHECK-SAME: : !_tupleToStruct1
// CHECK: p4hir.return %[[STRUCT]] : !_tupleToStruct1
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @make_large_tuple(%x : !b32i, %y: !i32i, %z: !b32i) -> tuple<!b32i, !i32i, !b32i> {
        %t = p4hir.tuple(%x, %y, %z) : tuple<!b32i, !i32i, !b32i>
        p4hir.return %t : tuple<!b32i, !i32i, !b32i>
    }
}

// CHECK-LABEL: module
// CHECK-LABEL: p4hir.func @create_nested
// CHECK-SAME: (%arg0: !b32i, %arg1: !i32i, %arg2: !b32i) -> !_tupleToStruct2
// CHECK: %[[INNER_STRUCT:.*]] = p4hir.struct (%arg0, %arg1) : !_tupleToStruct
// CHECK: %[[OUTER_STRUCT:.*]] = p4hir.struct (%[[INNER_STRUCT]], %arg2) : !_tupleToStruct2
// CHECK: p4hir.return %[[OUTER_STRUCT]] : !_tupleToStruct2
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
// CHECK-SAME: (%arg0: !_tupleToStruct2) -> !i32i
// CHECK: %[[INNER_STRUCT:.*]] = p4hir.struct_extract %arg0["element_0"] : !_tupleToStruct2
// CHECK: %[[FIELD:.*]] = p4hir.struct_extract %[[INNER_STRUCT]]["element_1"] : !_tupleToStruct
// CHECK: p4hir.return %[[FIELD]] : !i32i
// CHECK-NOT: p4hir.tuple
// CHECK-NOT: p4hir.tuple_extract

module {
   p4hir.func @nested_extract(%t : tuple<tuple<!b32i, !i32i>, !b32i>) -> !i32i {
        %inner = p4hir.tuple_extract %t[0] : tuple<tuple<!b32i, !i32i>, !b32i>
        %e = p4hir.tuple_extract %inner[1] : tuple<!b32i, !i32i>
        p4hir.return %e : !i32i
    }
}