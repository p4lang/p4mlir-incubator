// RUN: p4mlir-opt --p4hir-remove-aliases %s | FileCheck %s

// Derived using: p4mlir-translate --typeinference-only third_party/p4c/testdata/p4_16_samples/newtype.p4

!anon = !p4hir.enum<NoAction>
!b32i = !p4hir.bit<32>
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!infint = !p4hir.infint
!string = !p4hir.string
!type_T = !p4hir.type_var<"T">
#exact = #p4hir.match_kind<"exact">
#in = #p4hir<dir in>
#out = #p4hir<dir out>
#undir = #p4hir<dir undir>
// CHECK-NOT: !p4hir.alias
!N32_ = !p4hir.alias<"N32", !b32i>
!t = !p4hir.struct<"t", hit: !p4hir.bool, miss: !p4hir.bool, action_run: !anon>
#int0_b32i = #p4hir.int<0> : !b32i
#int1_b32i = #p4hir.int<1> : !b32i
#int1_infint = #p4hir.int<1> : !infint
#int2_b32i = #p4hir.int<2> : !b32i
#int3_b32i = #p4hir.int<3> : !b32i
#int5_b32i = #p4hir.int<5> : !b32i
!NN32_ = !p4hir.alias<"NN32", !N32_>
!S = !p4hir.struct<"S", b: !b32i, n: !N32_>
!c = !p4hir.control<"c", (!p4hir.ref<!b32i>)>
!e = !p4hir.control<"e", (!p4hir.ref<!b32i>)>

// CHECK-LABEL: module
module {
  p4hir.extern @packet_in {
    p4hir.overload_set @extract {
      p4hir.func @extract_0<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "hdr"})
      p4hir.func @extract_1<!type_T>(!p4hir.ref<!type_T> {p4hir.dir = #out, p4hir.param_name = "variableSizeHeader"}, !b32i {p4hir.dir = #in, p4hir.param_name = "variableFieldSizeInBits"})
    }
    p4hir.func @lookahead<!type_T>() -> !type_T
    p4hir.func @advance(!b32i {p4hir.dir = #in, p4hir.param_name = "sizeInBits"})
    p4hir.func @length() -> !b32i
  }
  p4hir.extern @packet_out {
    p4hir.func @emit<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "hdr"})
  }
  p4hir.func @verify(!p4hir.bool {p4hir.dir = #in, p4hir.param_name = "check"}, !error {p4hir.dir = #in, p4hir.param_name = "toSignal"})
  p4hir.func action @NoAction() annotations {noWarn = "unused"} {
    p4hir.return
  }
  p4hir.overload_set @static_assert {
    p4hir.func @static_assert_0(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}, !string {p4hir.dir = #undir, p4hir.param_name = "message"}) -> !p4hir.bool
    p4hir.func @static_assert_1(!p4hir.bool {p4hir.dir = #undir, p4hir.param_name = "check"}) -> !p4hir.bool
  }
  p4hir.control @c(%arg0: !p4hir.ref<!b32i> {p4hir.dir = #out, p4hir.param_name = "x"})() {
    p4hir.control_local @__local_c_x_0 = %arg0 : !p4hir.ref<!b32i>
    %k = p4hir.variable ["k"] : <!N32_>
    p4hir.control_local @__local_c_k_0 = %k : !p4hir.ref<!N32_>
    %nn = p4hir.variable ["nn"] : <!NN32_>
    p4hir.control_local @__local_c_nn_0 = %nn : !p4hir.ref<!NN32_>
    p4hir.table @t {
      p4hir.table_actions {
        p4hir.table_action @NoAction() {
          p4hir.call @NoAction () : () -> ()
        }
      }
      p4hir.table_key {
        %val = p4hir.read %k : <!N32_>
        p4hir.match_key #exact %val : !N32_ annotations {name = "k"}
      }
      p4hir.table_default_action {
        p4hir.call @NoAction () : () -> ()
      }
    }
    p4hir.control_apply {
      %c0_b32i = p4hir.const #int0_b32i
      %cast = p4hir.cast(%c0_b32i : !b32i) : !b32i
      %b = p4hir.variable ["b", init] : <!b32i>
      p4hir.assign %cast, %b : <!b32i>
      %c1_b32i = p4hir.const #int1_b32i
      %cast_0 = p4hir.cast(%c1_b32i : !b32i) : !N32_
      %n = p4hir.variable ["n", init] : <!N32_>
      p4hir.assign %cast_0, %n : <!N32_>
      %n1 = p4hir.variable ["n1"] : <!N32_>
      %s = p4hir.variable ["s"] : <!S>
      %c5_b32i = p4hir.const #int5_b32i
      %cast_1 = p4hir.cast(%c5_b32i : !b32i) : !N32_
      %cast_2 = p4hir.cast(%cast_1 : !N32_) : !NN32_
      %n5 = p4hir.variable ["n5", init] : <!NN32_>
      p4hir.assign %cast_2, %n5 : <!NN32_>
      %val = p4hir.read %b : <!b32i>
      %cast_3 = p4hir.cast(%val : !b32i) : !N32_
      p4hir.assign %cast_3, %n : <!N32_>
      %val_4 = p4hir.read %n : <!N32_>
      %cast_5 = p4hir.cast(%val_4 : !N32_) : !NN32_
      p4hir.assign %cast_5, %nn : <!NN32_>
      %val_6 = p4hir.read %n : <!N32_>
      p4hir.assign %val_6, %k : <!N32_>
      %val_7 = p4hir.read %n : <!N32_>
      %cast_8 = p4hir.cast(%val_7 : !N32_) : !b32i
      p4hir.assign %cast_8, %arg0 : <!b32i>
      %c1 = p4hir.const #int1_infint
      %cast_9 = p4hir.cast(%c1 : !infint) : !b32i
      %cast_10 = p4hir.cast(%cast_9 : !b32i) : !N32_
      p4hir.assign %cast_10, %n1 : <!N32_>
      %val_11 = p4hir.read %n : <!N32_>
      %val_12 = p4hir.read %n1 : <!N32_>
      %eq = p4hir.cmp(eq, %val_11, %val_12) : !N32_, !p4hir.bool
      p4hir.if %eq {
        %c2_b32i = p4hir.const #int2_b32i
        %cast_21 = p4hir.cast(%c2_b32i : !b32i) : !b32i
        p4hir.assign %cast_21, %arg0 : <!b32i>
      }
      %b_field_ref = p4hir.struct_extract_ref %s["b"] : <!S>
      %val_13 = p4hir.read %b : <!b32i>
      p4hir.assign %val_13, %b_field_ref : <!b32i>
      %n_field_ref = p4hir.struct_extract_ref %s["n"] : <!S>
      %val_14 = p4hir.read %n : <!N32_>
      p4hir.assign %val_14, %n_field_ref : <!N32_>
      %t_apply_result = p4hir.table_apply @c::@t : !t
      %val_15 = p4hir.read %s : <!S>
      %b_16 = p4hir.struct_extract %val_15["b"] : !S
      %val_17 = p4hir.read %s : <!S>
      %n_18 = p4hir.struct_extract %val_17["n"] : !S
      %cast_19 = p4hir.cast(%n_18 : !N32_) : !b32i
      %eq_20 = p4hir.cmp(eq, %b_16, %cast_19) : !b32i, !p4hir.bool
      p4hir.if %eq_20 {
        %c3_b32i = p4hir.const #int3_b32i
        %cast_21 = p4hir.cast(%c3_b32i : !b32i) : !b32i
        p4hir.assign %cast_21, %arg0 : <!b32i>
      }
    }
  }
  p4hir.package @top("_e" : !e {p4hir.dir = #undir, p4hir.param_name = "_e"})
  %c = p4hir.construct @c () : !c
  p4hir.instantiate @top (%c : !c) as @main
}
