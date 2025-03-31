// RUN: p4mlir-opt --mlir-print-ir-after-failure --verify-roundtrip -debug-only=canonicalize %s | FileCheck %s

#true = #p4hir.bool<true> : !p4hir.bool
!int32i = !p4hir.int<32>

// CHECK: module
module {
  // creating params - First const is true，second is 2.
  %0 = p4hir.const #true
  %1 = p4hir.const #p4hir.int<2> : !int32i
  
  // creating array - First param is true，second is 2.
  %arr1 = p4hir.array_create %0, %1 : !p4hir.bool, !int32i -> !p4hir.array<!p4hir.bool, 2>
}
