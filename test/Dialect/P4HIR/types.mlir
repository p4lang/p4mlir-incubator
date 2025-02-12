// RUN: p4mlir-opt %s | FileCheck %s

!unknown = !p4hir.unknown
!error = !p4hir.error
!dontcare = !p4hir.dontcare
!bit42 = !p4hir.bit<42>
!ref = !p4hir.ref<!p4hir.bit<42>>
!void = !p4hir.void

!action_noparams = !p4hir.func<()>
!action_params = !p4hir.func<(!p4hir.int<42>, !ref, !p4hir.int<42>, !p4hir.bool)>

!struct = !p4hir.struct<"struct_name", boolfield : !p4hir.bool, bitfield : !bit42>
!nested_struct = !p4hir.struct<"another_name", neststructfield : !struct, bitfield : !bit42>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
}
