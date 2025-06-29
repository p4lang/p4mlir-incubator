// RUN: p4mlir-opt --p4hir-enum-elimination %s | FileCheck %s

// CHECK: ![[suits:.*]] = !p4hir.ser_enum<"Suits", !b32i, Clubs : #int0_b32i, Diamonds : #int1_b32i, Spades : #int2_b32i>
!Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Spades>

// CHECK: #[[diamond:.*]] = #p4hir.enum_field<Diamonds, ![[suits]]> : ![[suits]]
#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits

// CHECK-LABEL: module
module {
  // CHECK: p4hir.func @process_suit(%arg0: ![[suits]])
  p4hir.func @process_suit(%arg0: !Suits) {
    p4hir.return
  }

  // CHECK: %[[var:.*]] = p4hir.variable ["suit"] : <![[suits]]>
  %var = p4hir.variable ["suit"] : <!Suits>

  // CHECK: %[[const:.*]] = p4hir.const #[[diamond]]
  %c = p4hir.const #Suits_Diamonds

  // CHECK: p4hir.assign %[[const]], %[[var]] : <![[suits]]>
  p4hir.assign %c, %var : <!Suits>

  // CHECK: %[[val:.*]] = p4hir.read %[[var]] : <![[suits]]>
  %val = p4hir.read %var : <!Suits>

  // CHECK: %[[eq:.*]] = p4hir.cmp(eq, %[[val]], %[[const]]) : ![[suits]], !p4hir.bool
  %eq = p4hir.cmp(eq, %val, %c) : !Suits, !p4hir.bool

  // CHECK p4hir.call @process_suit(%[[const]]) : (![[suits]]) -> ()
  p4hir.call @process_suit(%val) : (!Suits) -> ()

}

