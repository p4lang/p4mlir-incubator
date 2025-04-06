// RUN: p4mlir-opt %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

!b32i = !p4hir.bit<32>
!infint = !p4hir.infint
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int9_infint = #p4hir.int<9> : !infint
#int10_infint = #p4hir.int<10> : !infint

// CHECK: module
module {
    %0 = p4hir.const #int0_b32i
    %cast = p4hir.cast(%0 : !b32i) : !b32i
    %sum = p4hir.variable ["sum", init] : <!b32i>
    p4hir.assign %cast, %sum : <!b32i>
    p4hir.scope {
        %1 = p4hir.const #int0_b32i
        %cast_1 = p4hir.cast(%1 : !b32i) : !b32i
        %i = p4hir.variable ["i", init] : <!b32i>
        p4hir.assign %cast_1, %i : <!b32i>
        p4hir.for : cond {
            %c10 = p4hir.const #int10_infint
            %cast_2 = p4hir.cast(%c10 : !infint) : !b32i
            %val_3 = p4hir.read %i : <!b32i>
            %cond = p4hir.cmp(lt, %val_3, %cast_2) : !b32i, !p4hir.bool
            p4hir.condition %cond
        } body {
            %c1_b32i = p4hir.const #int1_b32i
            %val_2 = p4hir.read %sum : <!b32i>
            %add = p4hir.binop(add, %val_2, %c1_b32i) : !b32i
            p4hir.assign %add, %sum : <!b32i>
            p4hir.yield
        } updates {
            %c1_b32i = p4hir.const #int1_b32i
            %val_2 = p4hir.read %i : <!b32i>
            %add = p4hir.binop(add, %val_2, %c1_b32i) : !b32i
            p4hir.assign %add, %i : <!b32i>
            p4hir.yield
        }
    }
}

// CHECK: module
module {
  %c0_b32i = p4hir.const #int0_b32i
  %cast = p4hir.cast(%c0_b32i : !b32i) : !b32i
  %sum = p4hir.variable ["sum", init] : <!b32i>
  p4hir.assign %cast, %sum : <!b32i>
  p4hir.scope {
    %c0_b32i_0 = p4hir.const #int0_b32i
    %cast_1 = p4hir.cast(%c0_b32i_0 : !b32i) : !b32i
    %c9 = p4hir.const #int9_infint
    %cast_2 = p4hir.cast(%c9 : !infint) : !b32i
    %range = p4hir.range(%cast_1, %cast_2) : !p4hir.set<!b32i>
    p4hir.foreach %arg0 : !b32i in %range : !p4hir.set<!b32i> annotations {unlikely} {
      %val_3 = p4hir.read %sum : <!b32i>
      %add = p4hir.binop(add, %val_3, %arg0) : !b32i
      p4hir.assign %add, %sum : <!b32i>
      p4hir.yield
    }
  }
}
