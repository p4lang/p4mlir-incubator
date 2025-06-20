// RUN: p4mlir-opt --p4hir-enum-elimination %s | FileCheck %s

!Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Spades>

#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits

// CHECK-LABEL: module
module {
  // p4hir.func @process_suit(%arg0: !Suits) {
  //   p4hir.return
  // }

  %c = p4hir.const #Suits_Diamonds

  // p4hir.call @process_suit(%c) : (!Suits) -> ()
}

