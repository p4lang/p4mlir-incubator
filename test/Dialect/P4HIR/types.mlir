// RUN: p4mlir-opt --verify-roundtrip  %s | FileCheck %s

!unknown = !p4hir.unknown
!error = !p4hir.error<ErrorA, ErrorB>
!dontcare = !p4hir.dontcare
!bit42 = !p4hir.bit<42>
!ref = !p4hir.ref<!p4hir.bit<42>>
!void = !p4hir.void
!string = !p4hir.string

!action_noparams = !p4hir.func<()>
!action_params = !p4hir.func<(!p4hir.int<42>, !ref, !p4hir.int<42>, !p4hir.bool)>

!struct = !p4hir.struct<"struct_name", boolfield : !p4hir.bool, bitfield : !bit42>
!nested_struct = !p4hir.struct<"another_name", neststructfield : !struct, bitfield : !bit42>

!Suits = !p4hir.enum<"Suits", Clubs, Diamonds, Hearths, Spades>

!PortId_t = !p4hir.alias<"PortId_t", !bit42>

#b1 = #p4hir.int<1> : !bit42
#b2 = #p4hir.int<2> : !bit42
#b3 = #p4hir.int<3> : !bit42
#b4 = #p4hir.int<4> : !bit42

!SuitsSerializable = !p4hir.ser_enum<"Suits", !bit42, Clubs : #b1, Diamonds : #b2, Hearths : #b3, Spades : #b4>

!validity = !p4hir.validity.bit
!header = !p4hir.header<"struct_name", boolfield : !p4hir.bool, bitfield : !bit42, __validity : !validity>

!b9i = !p4hir.bit<9>
!M = !p4hir.struct<"M", e: !PortId_t, es: !b9i>

#valid = #p4hir<validity.bit valid>
#invalid = #p4hir<validity.bit invalid>

!Narrow = !p4hir.alias<"Narrow", !b9i>
#int192_Narrow = #p4hir.int<192> : !Narrow

!tuple = tuple<!bit42, !void, !SuitsSerializable>

!HeaderWithVarbit = !p4hir.header<"HeaderWithVarbit", varbitdata: !p4hir.varbit<32>>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
}
