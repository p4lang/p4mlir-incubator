// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

enum bit<16> EthTypes {
    IPv4 = 0x0800,
    ARP = 0x0806,
    RARP = 0x8035,
    EtherTalk = 0x809B,
    VLAN = 0x8100,
    IPX = 0x8137,
    IPv6 = 0x86DD
}

struct Ethernet {
    bit<48> src;
    bit<48> dest;
    EthTypes type;
}

struct Headers {
    Ethernet eth;
}

// CHECK: !EthTypes = !p4hir.ser_enum<"EthTypes", !b16i, ARP : #int2054_b16i, EtherTalk : #int-32613_b16i, IPX : #int-32457_b16i, IPv4 : #int2048_b16i, IPv6 : #int-31011_b16i, RARP : #int-32715_b16i, VLAN : #int-32512_b16i>
// CHECK: !Ethernet = !p4hir.struct<"Ethernet", src: !b48i, dest: !b48i, type: !EthTypes>
// CHECK: #EthTypes_IPv4_ = #p4hir.enum_field<IPv4, !EthTypes> : !EthTypes
// CHECK-LABEL: module

// CHECK-LABEL: p4hir.func action @test(%arg0: !p4hir.ref<!Headers>
// CHECK: p4hir.const #EthTypes_IPv4_
action test(inout Headers h) {
    if (h.eth.type == EthTypes.IPv4)
         h.eth.src = h.eth.dest;
    else
         h.eth.type = (EthTypes)(bit<16>)0;
}
