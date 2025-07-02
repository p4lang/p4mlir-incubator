// RUN: p4mlir-opt --p4hir-ser-enum-elimination %s | FileCheck %s

!b32i = !p4hir.bit<32>

#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i

// CHECK-NOT: !p4hir.ser_enum
!Suits = !p4hir.ser_enum<"Suits", !b32i, Clubs : #int0_b32i, Diamonds : #int1_b32i, Spades : #int2_b32i>

// CHECK-NOT: !p4hir.enum_field
#Suits_Clubs = #p4hir.enum_field<Clubs, !Suits> : !Suits
#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits
#Suits_Spades = #p4hir.enum_field<Spades, !Suits> : !Suits

!T = !p4hir.struct<"T", t1: !b32i, t2: !Suits>

// CHECK-LABEL: module
module {
  // CHECK: p4hir.func @process_suit(%[[arg:.*]]: !b32i)
  // CHECK: p4hir.switch (%[[arg]] : !b32i)
  // CHECK: p4hir.case(anyof, [#int0_b32i, #int2_b32i])
  // CHECK: %[[c1:.*]] = p4hir.const ["test"] #int1_b32i annotations {hidden}
  // CHECK: p4hir.call @process_suit (%[[c1]]) : (!b32i) -> ()
  // CHECK: p4hir.case(equal, [#int1_b32i])
  p4hir.func @process_suit(%arg0: !Suits) {
    p4hir.switch (%arg0 : !Suits) {
      p4hir.case(anyof, [#Suits_Clubs, #Suits_Spades]) {
        %c1 = p4hir.const ["test"] #Suits_Diamonds annotations {hidden}
        p4hir.call @process_suit(%c1) : (!Suits) -> ()
        p4hir.yield
      }
      p4hir.case(equal, [#Suits_Diamonds]) {
        p4hir.yield
      }
      p4hir.yield
    }
    p4hir.return
  }

  // CHECK: %[[var:.*]] = p4hir.variable ["suit"] : <!b32i>
  %var = p4hir.variable ["suit"] : <!Suits>

  // CHECK: %[[c2:.*]] = p4hir.const #int1_b32i
  %c2 = p4hir.const #Suits_Diamonds

  // CHECK: p4hir.assign %[[c2]], %[[var]] : <!b32i>
  p4hir.assign %c2, %var : <!Suits>

  // CHECK: %[[val:.*]] = p4hir.read %[[var]] : <!b32i>
  %val = p4hir.read %var : <!Suits>

  // CHECK: %[[eq:.*]] = p4hir.cmp(eq, %[[val]], %[[c2]]) : !b32i, !p4hir.bool
  %eq = p4hir.cmp(eq, %val, %c2) : !Suits, !p4hir.bool

  %t = p4hir.const ["t"] #p4hir.aggregate<[#int1_b32i, #Suits_Clubs]> : !T
  %t2 = p4hir.struct_extract %t["t2"] : !T

  // CHECK p4hir.call @process_suit(%[[c2]]) : (!b32i) -> ()
  p4hir.call @process_suit(%val) : (!Suits) -> ()
}

