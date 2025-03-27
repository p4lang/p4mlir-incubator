// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i8i = !p4hir.int<8>
!b8i = !p4hir.bit<8>
!infint = !p4hir.infint
#false = #p4hir.bool<false> : !p4hir.bool
#true = #p4hir.bool<true> : !p4hir.bool

#int-128_i8i = #p4hir.int<-128> : !i8i
#int-42_infint = #p4hir.int<-42> : !infint

!bit42 = !p4hir.bit<42>
#b1 = #p4hir.int<1> : !bit42
#b2 = #p4hir.int<2> : !bit42
#b3 = #p4hir.int<3> : !bit42
#b4 = #p4hir.int<4> : !bit42

!SuitsSerializable = !p4hir.ser_enum<"Suits", !bit42, Clubs : #b1, Diamonds : #b2, Hearths : #b3, Spades : #b4>
#Suits_Clubs = #p4hir.enum_field<Clubs, !SuitsSerializable> : !SuitsSerializable

// CHECK-LABEL: module
module {
  // CHECK: %[[cm42_i8i:.*]] = p4hir.const #int-42_i8i
  // CHECK: %[[c1_i8i:.*]] = p4hir.const #int1_i8i
  // CHECK: %[[c0_i8i:.*]] = p4hir.const #int0_i8i
  // CHECK: %[[cm128_i8i:.*]] = p4hir.const #int-128_i8i
  
  p4hir.func @blackhole(!i8i)

  %c-128_i8i = p4hir.const #int-128_i8i
  %false = p4hir.const #false
  %true = p4hir.const #true

  %c-42 = p4hir.const #int-42_infint

  %cast1 = p4hir.cast(%c-128_i8i : !i8i) : !i8i
  p4hir.call @blackhole(%cast1) : (!i8i) -> ()

  // CHECK: p4hir.call @blackhole (%[[cm128_i8i]]) : (!i8i) -> ()
  
  %cast2 = p4hir.cast(%false : !p4hir.bool) : !i8i
  p4hir.call @blackhole(%cast2) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole (%c0_i8i) : (!i8i) -> ()
  
  %cast3 = p4hir.cast(%true : !p4hir.bool) : !i8i
  p4hir.call @blackhole(%cast3) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole (%[[c1_i8i]]) : (!i8i) -> ()

  %cast4 = p4hir.cast(%c-42 : !p4hir.infint) : !i8i
  p4hir.call @blackhole(%cast4) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole (%[[cm42_i8i]]) : (!i8i) -> ()

  %castA = p4hir.cast(%cast1 : !i8i) : !b8i
  %castB = p4hir.cast(%castA : !b8i) : !i8i
  p4hir.call @blackhole(%castB) : (!i8i) -> ()

  %Suits_Clubs = p4hir.const #Suits_Clubs
  %cast5 = p4hir.cast(%Suits_Clubs : !SuitsSerializable) : !i8i
  p4hir.call @blackhole(%cast5) : (!i8i) -> ()
}
