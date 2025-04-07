// RUN: p4mlir-opt %s | FileCheck %s

!type_H1_ = !p4hir.type_var<"H1">
!type_M1_ = !p4hir.type_var<"M1">  
!Pipeline_type_H1_type_M1_ = !p4hir.package<"Pipeline"<!type_H1_, !type_M1_>>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %uninitialized = p4hir.uninitialized : !p4hir.bool
  %uninitialized2 = p4hir.uninitialized : !Pipeline_type_H1_type_M1_
}
