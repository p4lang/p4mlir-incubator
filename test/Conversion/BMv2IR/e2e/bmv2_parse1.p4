// This a temporary test, it's main purpose is to check that the bmv2-pipeline runs without errors
// TODO: remove this test once we have more robust e2e testing
// RUN: p4mlir-translate --target bmv2 --arch v1model --std p4-16 ~/work/p4/basic1/bmv2_parse1.p4 &> %t_1.mlir
// RUN: p4mlir-opt -p='builtin.module(bmv2-pipeline)' %t_1.mlir -o %t_2.mlir
// RUN: p4mlir-to-json --p4hir-to-bmv2-json %t_2.mlir -o - | FileCheck %s
#include <core.p4>
#include <v1model.p4>

header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> eth_type;
}

header H {
    bit<8> a;
    bit<8> b;
}

struct headers {
    ethernet_t eth_hdr;
    H h;
}

struct Meta {}

parser p(packet_in pkt, out headers hdr, inout Meta m, inout standard_metadata_t sm) {
    state start {
        transition parse_eth;
    }
    state parse_eth {
        pkt.extract(hdr.eth_hdr);
        transition select(sm.ingress_port) {
            255: parse_h;
            default: accept;
        }
    }
    state parse_h {
        pkt.extract(hdr.h);
        transition accept;
    }
}

control ingress(inout headers h, inout Meta m, inout standard_metadata_t sm) {
    apply {
        h.eth_hdr.dst_addr =  0xFFFFFFFFFFFF;
        h.eth_hdr.src_addr =  0xFFFFFFFFFFFF;
        h.eth_hdr.eth_type =  0xFFFF;

        if (h.h.isValid()) {
            h.h.a  = 0xA;
            h.h.b  = 0xB;
        }
    }
}

control vrfy(inout headers h, inout Meta m) { apply {} }

control update(inout headers h, inout Meta m) { apply {} }

control egress(inout headers h, inout Meta m, inout standard_metadata_t sm) { apply {} }

control deparser(packet_out pkt, in headers h) {
    apply {
        pkt.emit(h);
    }
}
V1Switch(p(), vrfy(), ingress(), egress(), update(), deparser()) main;


// CHECK: {
// CHECK:   "actions": [
// CHECK:   "calculations": [],
// CHECK:   "checksums": [],
// CHECK:   "deparsers": [
// CHECK:   "header_types": [
// CHECK:   "headers": [
// CHECK:   "parsers": [
// CHECK:   "pipelines": [
// CHECK: }
