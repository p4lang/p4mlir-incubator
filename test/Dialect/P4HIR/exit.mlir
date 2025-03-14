// RUN: p4mlir-opt %s | FileCheck %s

// CHECK: module
module {
  func @test_exit_simple() {
    p4hir.exit
  }

  func @test_exit_in_condition() {
    %ttl = p4hir.const #p4hir.int<0> : !p4hir.int<8>
    %cond = p4hir.cmp(eq, %ttl, 0)
    p4hir.cond_br %cond, ^exit, ^continue
    
  ^exit:
    p4hir.exit
  ^continue:
    p4hir.return
  }

  func @test_exit_nested() {
    %ttl = p4hir.const #p4hir.int<1> : !p4hir.int<8>
    p4hir.if %ttl {
      p4hir.exit
    } else {
      %new_ttl = p4hir.const #p4hir.int<5> : !p4hir.int<8>
      p4hir.return %new_ttl : !p4hir.int<8>
    }
  }
}