// RUN: p4mlir-opt --p4hir-flatten-cfg %s | FileCheck %s

module {
  // CHECK-LABEL: @ifthen
  // CHECK-NOT: p4hir.if
  p4hir.func @ifthen() {
    // CHECK: %[[false:.*]] = p4hir.const #false
    // CHECK: p4hir.cond_br %[[false]] ^[[bb1:.*]], ^[[bb2:.*]]
    // CHECK:  ^[[bb1]]:
    // CHECK:   p4hir.const #true
    // CHECK:   p4hir.br ^[[bb2]]
    // CHECK: ^[[bb2]]:
    // CHECK:  p4hir.return
    %a = p4hir.variable ["a", init] : <!p4hir.bool>
    %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool
    p4hir.if %0 {
      %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
      p4hir.assign %29, %a :  <!p4hir.bool>
    }
    p4hir.return
  }   

  // CHECK-LABEL: @ifthenelse
  // CHECK-NOT: p4hir.if
  p4hir.func @ifthenelse() {
    // CHECK: %[[false:.*]] = p4hir.const #false
    // CHECK: p4hir.cond_br %[[false]] ^[[bb1:.*]], ^[[bb2:.*]]
    // CHECK:  ^[[bb1]]:
    // CHECK:   p4hir.const #true
    // CHECK:   p4hir.br ^[[bb3:.*]]
    // CHECK: ^[[bb2]]:
    // CHECK:   p4hir.const #false
    // CHECK:   p4hir.br ^[[bb3]]
    // CHECK: ^[[bb3]]:    
    // CHECK:  p4hir.return
    %a = p4hir.variable ["a", init] : <!p4hir.bool>
    %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool  
    p4hir.if %0 {
      %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
      p4hir.assign %29, %a :  <!p4hir.bool>
    } else {
      %29 = p4hir.const #p4hir.bool<false> : !p4hir.bool
      p4hir.assign %29, %a :  <!p4hir.bool>
    }
    p4hir.return
  }

  // CHECK-LABEL: @ternary.bool
  // CHECK-NOT: p4hir.ternary
  p4hir.func @ternary.bool() -> !p4hir.bool {
    // CHECK: %[[false:.*]] = p4hir.const #false
    // CHECK: p4hir.cond_br %[[false]] ^[[bb1:.*]], ^[[bb2:.*]]
    // CHECK: ^[[bb1]]:
    // CHECK:   %[[true:.*]] = p4hir.const #true
    // CHECK:   p4hir.br ^[[bb3:.*]](%[[true]] : !p4hir.bool)
    // CHECK: ^[[bb2]]:
    // CHECK:   %[[false2:.*]] = p4hir.const #false
    // CHECK:   p4hir.br ^[[bb3]](%[[false2]] : !p4hir.bool)
    // CHECK: ^[[bb3]](%0: !p4hir.bool):
    
    %0 = p4hir.const #p4hir.bool<false> : !p4hir.bool
    %1 = p4hir.if %0 -> !p4hir.bool {
      %29 = p4hir.const #p4hir.bool<true> : !p4hir.bool
      p4hir.yield %29 : !p4hir.bool
    } else {
      %29 = p4hir.const #p4hir.bool<false> : !p4hir.bool
      p4hir.yield %29 : !p4hir.bool
    }
    p4hir.return %1 : !p4hir.bool
  }

  // CHECK-LABEL: @ternary.int
  // CHECK-NOT: p4hir.ternary

  p4hir.func @ternary.int()  -> !p4hir.int<32>{
    // CHECKL %[[false:.*]] = p4hir.const #false
    // CHECK: p4hir.cond_br %[[false]] ^[[bb1:.*]], ^[[bb2:.*]]
    // CHECK: ^[[bb1]]:
    // CHECK:   %[[truec:.*]] = p4hir.const #int42_i32i
    // CHECK:   p4hir.br ^[[bb3:.*]](%[[truec]] : !i32i)
    // CHECK: ^[[bb2]]:
    // CHECK:   %[[falsec:.*]] = p4hir.const #int100500_i32i
    // CHECK:   p4hir.br ^[[bb3]](%[[falsec]] : !i32i)
    // CHECK: ^[[bb3]](%0: !i32i):
  
    %1 = p4hir.const #p4hir.bool<false> : !p4hir.bool  
    %2 = p4hir.if %1 -> !p4hir.int<32> {
      %29 = p4hir.const #p4hir.int<42> : !p4hir.int<32>
      p4hir.yield %29 : !p4hir.int<32>
    } else {
      %29 = p4hir.const #p4hir.int<100500> : !p4hir.int<32>
      p4hir.yield %29 : !p4hir.int<32>
    }
    p4hir.return %2 : !p4hir.int<32>
  }
}
