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

  // CHECK: %[[const:.*]] = p4hir.const #[[diamond]]
  %c = p4hir.const #Suits_Diamonds

  // CHECK p4hir.call @process_suit(%[[const]]) : (![[suits]]) -> ()
  p4hir.call @process_suit(%c) : (!Suits) -> ()
}

