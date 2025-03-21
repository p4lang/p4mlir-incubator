// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

// CHECK-DAG: !PortId_32_t = !p4hir.alias<"PortId_32_t" annotations {p4runtime_translation = ["p4.org/psa/v1/PortId_32_t", ",", "32"]}, !b9i>

@p4runtime_translation("p4.org/psa/v1/PortId_32_t", 32)
type bit<9> PortId_32_t;

// CHECK-DAG: !packet_in_header_t = !p4hir.header<"packet_in_header_t" {controller_header = ["packet_in"]}, ingress_port: !PortId_32_t {id = ["1"]}, target_egress_port: !PortId_32_t {id = ["2"]}, __valid: !validity_bit>
@controller_header("packet_in") header packet_in_header_t {
    @id(1)
    PortId_32_t ingress_port;
    @id(2)
    PortId_32_t target_egress_port;
}

// CHECK-DAG: !PreservedFieldList = !p4hir.ser_enum<"PreservedFieldList" {p4runtime_translation = ["p4.org/psa/v1/foo", ",", "enum"]}, !b8i, Field : #int1_b8i>
@p4runtime_translation("p4.org/psa/v1/foo", enum)
enum bit<8> PreservedFieldList {
    Field = 8w1
}

// CHECK-DAG: !SomeEnum = !p4hir.enum<"SomeEnum" {p4runtime_translation = ["p4.org/psa/v1/bar", ",", "enum"]}, Field, Field2>
@p4runtime_translation("p4.org/psa/v1/bar", enum)
enum SomeEnum {
  Field, Field2
}

// CHECK-DAG: #PreservedFieldList_Field = #p4hir.enum_field<Field, !PreservedFieldList> : !PreservedFieldList
// CHECK-DAG: !Meta = !p4hir.struct<"Meta" {controller_header = ["foo"]}, b: !b1i {field_list = #PreservedFieldList_Field}, f: !PreservedFieldList>

@controller_header("foo")
struct Meta {
    @field_list(PreservedFieldList.Field)
    bit<1> b;
    PreservedFieldList f;
}

action foo(in Meta m, @optional in packet_in_header_t hdr, SomeEnum e) {}
