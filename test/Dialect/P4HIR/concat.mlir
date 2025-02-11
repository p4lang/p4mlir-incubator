// RUN: p4mlir-opt %s | FileCheck %s

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
  %bit5  = p4hir.const #p4hir.int<2> : !p4hir.bit<5>
  %bit10 = p4hir.const #p4hir.int<4> : !p4hir.bit<10>
  %int5  = p4hir.const #p4hir.int<6> : !p4hir.int<5>
  %int10 = p4hir.const #p4hir.int<8> : !p4hir.int<10>

  %0 = p4hir.concat(%bit5 : !p4hir.bit<5>, %bit5 :  !p4hir.bit<5>)  : !p4hir.bit<10>
  %1 = p4hir.concat(%bit5 : !p4hir.bit<5>, %bit10 : !p4hir.bit<10>) : !p4hir.bit<15>

  %2 = p4hir.concat(%int5 : !p4hir.int<5>, %int5 :  !p4hir.int<5>)  : !p4hir.int<10>
  %3 = p4hir.concat(%int5 : !p4hir.int<5>, %int10 : !p4hir.int<10>) : !p4hir.int<15>

  %4 = p4hir.concat(%bit5 : !p4hir.bit<5>, %int5 :  !p4hir.int<5>)  : !p4hir.bit<10>
  %5 = p4hir.concat(%bit5 : !p4hir.bit<5>, %int10 : !p4hir.int<10>) : !p4hir.bit<15>

  %6 = p4hir.concat(%int5 : !p4hir.int<5>, %bit5 :  !p4hir.bit<5>)  : !p4hir.int<10>
  %7 = p4hir.concat(%int5 : !p4hir.int<5>, %bit10 : !p4hir.bit<10>) : !p4hir.int<15>
}
