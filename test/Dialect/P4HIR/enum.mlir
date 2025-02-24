// RUN: p4mlir-opt %s | FileCheck %s

!Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Hearths, Spades>

#Suits_Clubs = #p4hir.enum_field<Clubs, !Suits> : !Suits
#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits

// CHECK: module
module {
  %Suits_Diamonds = p4hir.const #Suits_Diamonds
}
