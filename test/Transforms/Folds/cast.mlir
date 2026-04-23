// RUN: p4mlir-opt --canonicalize %s | FileCheck %s

!bool = !p4hir.bool
!b1i = !p4hir.bit<1>
!i8i = !p4hir.int<8>
!i16i = !p4hir.int<16>
!b8i = !p4hir.bit<8>
!b16i = !p4hir.bit<16>
!b32i = !p4hir.bit<32>
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

  // ---- Cast chain composition safety tests ----

  p4hir.func @blackhole_bool(!bool)
  p4hir.func @blackhole_b1i(!b1i)
  p4hir.func @blackhole_b8i(!b8i)
  p4hir.func @blackhole_b16i(!b16i)
  p4hir.func @blackhole_b32i(!b32i)
  p4hir.func @blackhole_b42i(!bit42)
  p4hir.func @blackhole_SuitsSerializable(!SuitsSerializable)

  // Use non-constant arguments so chains are not constant-folded away.
  p4hir.func @cast_chain_tests(%arg_b8 : !b8i, %arg_b16 : !b16i, %arg_i8 : !i8i, %arg_b32 : !b32i, %arg_b1 : !b1i, %arg_bool : !bool, %arg_serenum : !SuitsSerializable, %arg_b42 : !bit42) {
    // Safe: widen then reinterpret (w_B >= w_C).
    // bit<8> -> bit<16> -> int<16> folds to bit<8> -> int<16>.
    // CHECK-LABEL: @cast_chain_tests
    // CHECK: %[[V0:.*]] = p4hir.cast(%arg0 : !b8i) : !i16i
    // CHECK: p4hir.call @blackhole_i16i (%[[V0]])
    %c0 = p4hir.cast(%arg_b8 : !b8i) : !b16i
    %c1 = p4hir.cast(%c0 : !b16i) : !i16i
    p4hir.call @blackhole_i16i(%c1) : (!i16i) -> ()

    // Unsafe: reinterpret then widen (sign change before widen).
    // bit<8> -> int<8> -> int<16> must NOT fold: the chain sign-extends,
    // but a direct bit<8> -> int<16> would zero-extend.
    // CHECK: %[[V1:.*]] = p4hir.cast(%arg0 : !b8i) : !i8i
    // CHECK: %[[V2:.*]] = p4hir.cast(%[[V1]] : !i8i) : !i16i
    // CHECK: p4hir.call @blackhole_i16i (%[[V2]])
    %c2 = p4hir.cast(%arg_b8 : !b8i) : !i8i
    %c3 = p4hir.cast(%c2 : !i8i) : !i16i
    p4hir.call @blackhole_i16i(%c3) : (!i16i) -> ()

    // Safe: same-sign widen then widen (w_A <= w_B, s_A == s_B).
    // bit<8> -> bit<16> -> bit<32> folds to bit<8> -> bit<32>.
    // CHECK: %[[V3:.*]] = p4hir.cast(%arg0 : !b8i) : !b32i
    // CHECK: p4hir.call @blackhole_b32i (%[[V3]])
    %c4 = p4hir.cast(%arg_b8 : !b8i) : !b16i
    %c5 = p4hir.cast(%c4 : !b16i) : !b32i
    p4hir.call @blackhole_b32i(%c5) : (!b32i) -> ()

    // Unsafe: truncate then widen (w_A > w_B, w_B < w_C).
    // bit<16> -> bit<8> -> bit<32> must NOT fold: lossy truncation
    // followed by widening differs from the direct widen.
    // CHECK: %[[V4:.*]] = p4hir.cast(%arg1 : !b16i) : !b8i
    // CHECK: %[[V5:.*]] = p4hir.cast(%[[V4]] : !b8i) : !b32i
    // CHECK: p4hir.call @blackhole_b32i (%[[V5]])
    %c6 = p4hir.cast(%arg_b16 : !b16i) : !b8i
    %c7 = p4hir.cast(%c6 : !b8i) : !b32i
    p4hir.call @blackhole_b32i(%c7) : (!b32i) -> ()

    // Safe: widen then truncate (w_B >= w_C).
    // bit<8> -> bit<32> -> bit<16> folds to bit<8> -> bit<16>.
    // CHECK: %[[V6:.*]] = p4hir.cast(%arg0 : !b8i) : !b16i
    // CHECK: p4hir.call @blackhole_b16i (%[[V6]])
    %c8 = p4hir.cast(%arg_b8 : !b8i) : !b32i
    %c9 = p4hir.cast(%c8 : !b32i) : !b16i
    p4hir.call @blackhole_b16i(%c9) : (!b16i) -> ()

    // Safe: truncate then reinterpret (w_B >= w_C).
    // bit<16> -> bit<8> -> int<8> folds to bit<16> -> int<8>.
    // CHECK: %[[V7:.*]] = p4hir.cast(%arg1 : !b16i) : !i8i
    // CHECK: p4hir.call @blackhole_i8i (%[[V7]])
    %c10 = p4hir.cast(%arg_b16 : !b16i) : !b8i
    %c11 = p4hir.cast(%c10 : !b8i) : !i8i
    p4hir.call @blackhole_i8i(%c11) : (!i8i) -> ()

    // Safe: truncate then truncate (w_B >= w_C).
    // bit<32> -> bit<16> -> bit<8> folds to bit<32> -> bit<8>.
    // CHECK: %[[V8:.*]] = p4hir.cast(%arg3 : !b32i) : !b8i
    // CHECK: p4hir.call @blackhole_b8i (%[[V8]])
    %c12 = p4hir.cast(%arg_b32 : !b32i) : !b16i
    %c13 = p4hir.cast(%c12 : !b16i) : !b8i
    p4hir.call @blackhole_b8i(%c13) : (!b8i) -> ()

    // bit<1> -> bool -> bit<1> folds to source.
    // CHECK: p4hir.call @blackhole_b1i (%arg4) : (!b1i) -> ()
    %bool_1 = p4hir.cast(%arg_b1 : !b1i) : !bool
    %bit1_1 = p4hir.cast(%bool_1 : !bool) : !b1i
    p4hir.call @blackhole_b1i(%bit1_1) : (!b1i) -> ()

    // bool -> bit<1> -> bool folds to source.
    // CHECK: p4hir.call @blackhole_bool (%arg5) : (!p4hir.bool) -> ()
    %bit1_2 = p4hir.cast(%arg_bool : !bool) : !b1i
    %bool_2 = p4hir.cast(%bit1_2 : !b1i) : !bool
    p4hir.call @blackhole_bool(%bool_2) : (!bool) -> ()
    
    // ser_enum -> bits -> ser_enum folds to source.
    // CHECK: p4hir.call @blackhole_SuitsSerializable (%arg6) : (!Suits) -> ()
    %bit42_1 = p4hir.cast(%arg_serenum : !SuitsSerializable) : !bit42
    %suits_1 = p4hir.cast(%bit42_1 : !bit42) : !SuitsSerializable
    p4hir.call @blackhole_SuitsSerializable(%suits_1) : (!SuitsSerializable) -> ()

    // bits -> ser_enum -> bits folds to source.
    // CHECK: p4hir.call @blackhole_b42i (%arg7) : (!b42i) -> ()
    %suits_2 = p4hir.cast(%arg_b42 : !bit42) : !SuitsSerializable
    %bit42_2 = p4hir.cast(%suits_2 : !SuitsSerializable) : !bit42
    p4hir.call @blackhole_b42i(%bit42_2) : (!bit42) -> ()

    p4hir.return
  }
}
