// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!i8i = !p4hir.int<8>
!i16i = !p4hir.int<16>
!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
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
  // CHECK: %[[cm128_i16i:.*]] = p4hir.const #int-128_i16i
  // CHECK: %[[cm42_i8i:.*]] = p4hir.const #int-42_i8i
  // CHECK: %[[c1_i8i:.*]] = p4hir.const #int1_i8i
  // CHECK: %[[c0_i8i:.*]] = p4hir.const #int0_i8i
  // CHECK: %[[cm128_i8i:.*]] = p4hir.const #int-128_i8i
  
  p4hir.func @blackhole_i8i(!i8i)
  p4hir.func @blackhole_i16i(!i16i)

  %c-128_i8i = p4hir.const #int-128_i8i
  %false = p4hir.const #false
  %true = p4hir.const #true

  %c-42 = p4hir.const #int-42_infint

  %cast1 = p4hir.cast(%c-128_i8i : !i8i) : !i8i
  p4hir.call @blackhole_i8i(%cast1) : (!i8i) -> ()

  // CHECK: p4hir.call @blackhole_i8i (%[[cm128_i8i]]) : (!i8i) -> ()
  
  %cast2 = p4hir.cast(%false : !p4hir.bool) : !i8i
  p4hir.call @blackhole_i8i(%cast2) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole_i8i (%c0_i8i) : (!i8i) -> ()
  
  %cast3 = p4hir.cast(%true : !p4hir.bool) : !i8i
  p4hir.call @blackhole_i8i(%cast3) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole_i8i (%[[c1_i8i]]) : (!i8i) -> ()

  %cast4 = p4hir.cast(%c-42 : !p4hir.infint) : !i8i
  p4hir.call @blackhole_i8i(%cast4) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole_i8i (%[[cm42_i8i]]) : (!i8i) -> ()

  %castA = p4hir.cast(%cast1 : !i8i) : !b8i
  %castB = p4hir.cast(%castA : !b8i) : !i8i
  p4hir.call @blackhole_i8i(%castB) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole_i8i (%[[cm128_i8i]]) : (!i8i) -> ()

  %Suits_Clubs = p4hir.const #Suits_Clubs
  %cast5 = p4hir.cast(%Suits_Clubs : !SuitsSerializable) : !i8i
  p4hir.call @blackhole_i8i(%cast5) : (!i8i) -> ()
  // CHECK: p4hir.call @blackhole_i8i (%[[c1_i8i]]) : (!i8i) -> ()

  %cast6 = p4hir.cast(%c-128_i8i : !i8i) : !i16i
  p4hir.call @blackhole_i16i(%cast6) : (!i16i) -> ()
  // CHECK: p4hir.call @blackhole_i16i (%[[cm128_i16i]]) : (!i16i) -> ()

  // CHECK-LABEL: p4hir.func @cast_chain_compose_guard
  // CHECK: %[[SAFE:.*]] = p4hir.cast(%arg0 : !b8i) : !i16i
  // CHECK: p4hir.call @blackhole_i16i (%[[SAFE]]) : (!i16i) -> ()
  // CHECK: %[[UNSAFE_MID:.*]] = p4hir.cast(%arg0 : !b8i) : !i8i
  // CHECK: %[[UNSAFE:.*]] = p4hir.cast(%[[UNSAFE_MID]] : !i8i) : !i16i
  // CHECK: p4hir.call @blackhole_i16i (%[[UNSAFE]]) : (!i16i) -> ()
  // CHECK: p4hir.call @blackhole_i8i (%arg1) : (!i8i) -> ()
  p4hir.func @cast_chain_compose_guard(%arg0: !b8i, %arg1: !i8i) {
    %safe_mid = p4hir.cast(%arg0 : !b8i) : !b16i
    %safe = p4hir.cast(%safe_mid : !b16i) : !i16i
    p4hir.call @blackhole_i16i(%safe) : (!i16i) -> ()

    %unsafe_mid = p4hir.cast(%arg0 : !b8i) : !i8i
    %unsafe = p4hir.cast(%unsafe_mid : !i8i) : !i16i
    p4hir.call @blackhole_i16i(%unsafe) : (!i16i) -> ()

    %repr_mid = p4hir.cast(%arg1 : !i8i) : !b8i
    %repr = p4hir.cast(%repr_mid : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%repr) : (!i8i) -> ()

    p4hir.return
  }
}
