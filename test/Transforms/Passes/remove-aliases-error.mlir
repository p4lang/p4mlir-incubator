// RUN: p4mlir-opt --p4hir-remove-aliases %s -split-input-file -verify-diagnostics

!b32i = !p4hir.bit<32>
!N32 = !p4hir.alias<"N32", !b32i>

#int0_N32 = #p4hir.int<0> : !N32

// CHECK: module
module {
  p4hir.func @test(%arg0: !b32i) {
    p4hir.return
  }

  %c0_N32 = p4hir.const #int0_N32
  // expected-error@below {{'p4hir.call' op operand type mismatch: expected operand type '!p4hir.bit<32>', but provided '!p4hir.alias<"N32", !p4hir.bit<32>>' for operand number 0}}
  p4hir.call @test(%c0_N32) : (!N32) -> ()
}

// -----

!b32i = !p4hir.bit<32>
!N32 = !p4hir.alias<"N32", !b32i>

#int0_N32 = #p4hir.int<0> : !N32
#int1_b32i = #p4hir.int<1> : !b32i

// CHECK: module
module {
  %c1_b32i = p4hir.const #int1_b32i

  // expected-note@below {{prior use here}}
  %var = p4hir.variable ["x"] : <!N32>
  // expected-error@below {{use of value '%var' expects different type than prior uses: '!p4hir.alias<"N32", !p4hir.bit<32>>' vs '!p4hir.ref<!p4hir.alias<"N32", !p4hir.bit<32>>>'}}
  p4hir.assign %var, %c1_b32i : <!N32>
}

// -----

!b32i = !p4hir.bit<32>
!N32 = !p4hir.alias<"N32", !b32i>
!T = !p4hir.struct<"T", t1: !b32i, t2: !N32>

#int0_N32 = #p4hir.int<0> : !N32
#int1_b32i = #p4hir.int<1> : !b32i

// CHECK: module
module {
  // expected-error@below {{aggregate initializer type for struct field '"t2"' must match, expected: '!p4hir.alias<"N32", !p4hir.bit<32>>', got: '!p4hir.bit<32>'}}
  %agg = p4hir.const #p4hir.aggregate<[#int1_b32i, #int1_b32i]> : !T
}

