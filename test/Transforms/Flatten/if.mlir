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
}
