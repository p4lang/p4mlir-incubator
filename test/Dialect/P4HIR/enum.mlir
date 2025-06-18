// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Hearths, Spades>

#Suits_Clubs = #p4hir.enum_field<Clubs, !Suits> : !Suits
#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits

!anon = !p4hir.enum<Foo, Bar, Baz>
#anon_Foo = #p4hir.enum_field<Foo, !anon> : !anon

// CHECK: module
module {
  %Suits_Diamonds = p4hir.const #Suits_Diamonds
}
