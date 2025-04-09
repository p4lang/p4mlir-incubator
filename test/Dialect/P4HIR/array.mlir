// RUN: p4mlir-opt %s | FileCheck %s

!i32i = !p4hir.int<32>
!A = !p4hir.array<42 x !i32i>
!B = !p4hir.array<2 x !A>
!C = !p4hir.array<2 x !i32i>

#int10_i32i = #p4hir.int<10> : !i32i
#int20_i32i = #p4hir.int<20> : !i32i

// CHECK: module
module {
  %c = p4hir.const ["t"] #p4hir.aggregate<[#int10_i32i, #int20_i32i]> : !C

  p4hir.func action @test() {
    %vv = p4hir.variable ["vv"] : <!i32i>
    %val = p4hir.read %vv : <!i32i>
    %a = p4hir.array [%val, %val] : !C
    %idx = p4hir.const #p4hir.int<1> : !i32i
    %v1 = p4hir.array_get %c[%idx] : !C, !i32i
    p4hir.array_set %c[%idx] = %idx : !C, !i32i

    p4hir.return    
  }
}
