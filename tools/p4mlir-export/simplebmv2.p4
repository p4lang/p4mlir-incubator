// Simplest p4 program to be processed, which can go through
// p4c and be accepted bmv2 simple switch v1 model

#include <core.p4>
#include <v1model.p4>

// Typical BMv2 constant for ingress port
const bit<9> CPU_PORT = 255;

// Header type definition
struct headers {
    // Empty for minimal example
}

// Metadata structure
struct metadata {
    bit<9> ingress_port;
}

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_meta) {
    state start {
        meta.ingress_port = standard_meta.ingress_port;
        transition accept;
    }
}

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_meta) {
    apply {
        if (standard_meta.ingress_port == CPU_PORT) {
            // Special handling for CPU port
            standard_meta.egress_spec = CPU_PORT;
        }
    }
}

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_meta) {
    apply { }
}

control MyVerifyChecksum(inout headers hdr, 
                         inout metadata meta) {
    apply { }
}

control MyComputeChecksum(inout headers hdr, 
                          inout metadata meta) {
    apply { }
}

control MyDeparser(packet_out packet, 
                   in headers hdr) {
    apply { }
}

V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;