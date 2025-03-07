// RUN: p4mlir-opt %s | FileCheck %s

!u32i = !p4hir.bit<32>
!s32i = !p4hir.int<32>
!int = !p4hir.infint
!u8i  = !p4hir.bit<8>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module  {
  %u32ival = p4hir.const #p4hir.int<1> : !u32i
  %s32ival = p4hir.const #p4hir.int<1> : !s32i
  %intval = p4hir.const #p4hir.int<1> : !int

  %u8ishift = p4hir.const #p4hir.int<1> : !u8i
  %intshift = p4hir.const #p4hir.int<1> : !int

  // shl
  %u32i_shl_u8i = p4hir.shl(%u32ival : !u32i, %u8ishift : !u8i) : !u32i
  %u32i_shl_int = p4hir.shl(%u32ival : !u32i, %intshift : !int) : !u32i

  %s32i_shl_u8i = p4hir.shl(%s32ival : !s32i, %u8ishift : !u8i) : !s32i
  %s32i_shl_int = p4hir.shl(%s32ival : !s32i, %intshift : !int) : !s32i

  %int_shl_int = p4hir.shl(%intval : !int, %intshift : !int) : !int

  // shr
  %u32i_shr_u8i = p4hir.shr(%u32ival : !u32i, %u8ishift : !u8i) : !u32i
  %u32i_shr_int = p4hir.shr(%u32ival : !u32i, %intshift : !int) : !u32i

  %s32i_shr_u8i = p4hir.shr(%s32ival : !s32i, %u8ishift : !u8i) : !s32i
  %s32i_shr_int = p4hir.shr(%s32ival : !s32i, %intshift : !int) : !s32i

  %int_shr_int = p4hir.shr(%intval : !int, %intshift : !int) : !int
}
