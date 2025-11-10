// RUN: p4mlir-opt --p4hir-ser-enum-elimination %s | FileCheck %s

!b32i = !p4hir.bit<32>
!i32i = !p4hir.int<32>

#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int2_b32i = #p4hir.int<2> : !b32i

// CHECK-NOT: !p4hir.ser_enum
!Suits = !p4hir.ser_enum<"Suits", !b32i, Clubs : #int0_b32i, Diamonds : #int1_b32i, Spades : #int2_b32i>

// CHECK-NOT: !p4hir.enum_field
#Suits_Clubs = #p4hir.enum_field<Clubs, !Suits> : !Suits
#Suits_Diamonds = #p4hir.enum_field<Diamonds, !Suits> : !Suits
#Suits_Spades = #p4hir.enum_field<Spades, !Suits> : !Suits

// CHECK-NOT: !Suits
!T = !p4hir.struct<"T", t1: !b32i, t2: !Suits>

!validity_bit = !p4hir.validity.bit
#valid = #p4hir<validity.bit valid> : !validity_bit
!H = !p4hir.header<"H", x: !b32i, suits: !Suits, __valid: !validity_bit>
!H2 = !p4hir.header<"H2", x: !b32i, nested: !T, __valid: !validity_bit>

!A = !p4hir.array<2 x !T>

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

  // CHECK: %[[eq:.*]] = p4hir.cmp(eq, %[[val]] : !b32i, %[[c2]] : !b32i)
  %eq = p4hir.cmp(eq, %val : !Suits, %c2 : !Suits)

  // Test structs & extracts
  // CHECK:   #p4hir.aggregate<[#int1_b32i, #int0_b32i]> : !T
  %t = p4hir.const ["t"] #p4hir.aggregate<[#int1_b32i, #Suits_Clubs]> : !T
  %t2 = p4hir.struct_extract %t["t2"] : !T

  // Test headers
  %h = p4hir.variable ["h"] : <!H>
  %c0_b32i = p4hir.const #int0_b32i
  %valid = p4hir.const #valid
  // CHECK: p4hir.struct (%{{.*}}, %[[c2]], %{{.*}}) : !H
  %hdr_H = p4hir.struct (%c0_b32i, %c2, %valid) : !H
  p4hir.assign %hdr_H, %h : <!H>
  %serenum_field_ref = p4hir.struct_extract_ref %h["suits"] : <!H>
  %h2 = p4hir.variable ["h2"] : <!H2>

  // Test arrays
  // CHECK: #p4hir.aggregate<[#p4hir.aggregate<[#int1_b32i, #int0_b32i]> : !T, #p4hir.aggregate<[#int2_b32i, #int2_b32i]> : !T]> : !arr_2xT
  %c = p4hir.const ["t"] #p4hir.aggregate<[#p4hir.aggregate<[#int1_b32i, #Suits_Clubs]> : !T, #p4hir.aggregate<[#int2_b32i, #Suits_Spades]> : !T]> : !A
  %vvt = p4hir.variable ["vv"] : <!T>
  %valt = p4hir.read %vvt : <!T>
  %a = p4hir.array [%valt, %valt] : !A
  %aa = p4hir.variable ["aa"] : <!A>
  %idx = p4hir.const #p4hir.int<1> : !i32i
  %v1 = p4hir.array_get %c[%idx] : !A, !i32i
  %aa_ref = p4hir.array_element_ref %aa[%idx] : !p4hir.ref<!A>, !i32i
  p4hir.assign %v1, %aa_ref : <!T>

  // CHECK p4hir.call @process_suit(%[[c2]]) : (!b32i) -> ()
  p4hir.call @process_suit(%val) : (!Suits) -> ()
}

