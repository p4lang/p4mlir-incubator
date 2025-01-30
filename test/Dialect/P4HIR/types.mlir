// RUN: p4mlir-opt %s | FileCheck %s

!unknown = !p4hir.unknown
!error = !p4hir.error
!dontcare = !p4hir.dontcare
!ref = !p4hir.ref<!p4hir.bit<42>>

!action_noparams = !p4hir.action<()>
!action_params = !p4hir.action<(!p4hir.int<42>, !ref, !p4hir.int<42>, !p4hir.bool)>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
}
