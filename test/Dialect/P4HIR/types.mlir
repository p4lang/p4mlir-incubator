// RUN: p4mlir-opt %s | FileCheck %s

!unknown = !p4hir.unknown
!error = !p4hir.error
!dontcare = !p4hir.dontcare
!ref = !p4hir.ref<!p4hir.bit<42>>

!param_out = !p4hir.param<!ref, out>
!param_inout = !p4hir.param<!p4hir.ref<!p4hir.int<42>>, inout>
!param_in = !p4hir.param<!p4hir.int<42>, in>
!param_undir = !p4hir.param<!p4hir.int<42>>

!action_noparams = !p4hir.action<()>
!action_undirparams = !p4hir.action<(!p4hir.int<42>)>
!action_allparams = !p4hir.action<(in !p4hir.int<42>, out !p4hir.int<42>, inout !p4hir.int<42>, !p4hir.bool)>


// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
}
