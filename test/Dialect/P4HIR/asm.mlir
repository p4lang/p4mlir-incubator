// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!s32i = !p4hir.int<32>

// CHECK: module
module {
  p4hir.func action @simple(%arg0: !s32i) {
    %0 = p4hir.variable ["x", init] : <!s32i>
    p4hir.assign %arg0, %0 : <!s32i>

    p4hir.asm(
      out = [],
      in = [],
      in_out = [],
      {"" "~{flags}"}) -> !s32i
        
    p4hir.asm(
      out = [],
      in = [],
      in_out = [],
      {"xyz" "~{flags}"}) side_effects -> !s32i

    p4hir.asm(
      out = [%0 : !p4hir.ref<!s32i> (maybe_memory)],
      in = [],
      in_out = [%0 : !p4hir.ref<!s32i> (maybe_memory)],
      {"" "=*m,*m,~{flags}"}) side_effects -> !s32i

    p4hir.asm(
      out = [],
      in = [%0 : !p4hir.ref<!s32i> (maybe_memory)],
      in_out = [],
      {"" "*m,~{flags}"}) side_effects -> !s32i      

    p4hir.asm(
      out = [%0 : !p4hir.ref<!s32i> (maybe_memory)],
      in = [],
      in_out = [],
      {"" "=*m,~{flags}"}) side_effects -> !s32i
   
    p4hir.asm(
      out = [],
      in = [],
      in_out = [],
      {"" "=&r,=&r,1,~{flags}"}) side_effects -> !s32i
      
    p4hir.return
  }

}
