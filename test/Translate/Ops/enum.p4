// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | diff -u - %s.ref
// RUN: p4mlir-translate --typeinference-only --no-dump --Wdisable --dump-exported-p4 %s | p4test -

// CHECK: !Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Hearths, Spades>
// CHECK: #Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits
// CHECK: #Suits_Hearths = #p4hir.enum_field<Hearths, !Suits> : !Suits
// CHECK: #Suits_Spades = #p4hir.enum_field<Spades, !Suits> : !Suits

enum Suits { Clubs, Diamonds, Hearths, Spades }

// CHECK-LABEL: module
const Suits cEnum = Suits.Hearths;

// CHECK-LABEL: p4hir.func action @test
action test(inout bit<42> a, Suits b) {
  // CHECK: %Suits_Diamonds = p4hir.const #Suits_Diamonds
  // CHECK: %d = p4hir.variable ["d", init] : <!Suits>
  // CHECK: p4hir.assign %Suits_Diamonds, %d : <!Suits>
  // CHECK: p4hir.const ["cEnum"] #Suits_Hearths
  Suits d = Suits.Diamonds;
  if (b == Suits.Spades || b == cEnum) {
    a = a + 1;
  } else if (b == d) {
    a = a - 1;
  }
}
