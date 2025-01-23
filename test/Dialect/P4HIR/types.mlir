// RUN: p4mlir-opt %s | FileCheck %s

!unknown = !p4hir.unknown
!error = !p4hir.error
!dontcare = !p4hir.dontcare
!ref = !p4hir.ref<!p4hir.bit<42>>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
}
