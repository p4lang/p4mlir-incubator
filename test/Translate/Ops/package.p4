// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

extern packet_in {
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader,
                    in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(in bit<32> sizeInBits);
    bit<32> length();
}

extern packet_out {
    void emit<T>(in T hdr);
}

struct standard_metadata_t {
}

parser Parser<H, M>(packet_in b,
                    out H parsedHdr,
                    inout M meta,
                    inout standard_metadata_t standard_metadata);
control VerifyChecksum<H, M>(inout H hdr,
                             inout M meta);
control Ingress<H, M>(inout H hdr,
                      inout M meta,
                      inout standard_metadata_t standard_metadata);
control Egress<H, M>(inout H hdr,
                     inout M meta,
                     inout standard_metadata_t standard_metadata);
control ComputeChecksum<H, M>(inout H hdr,
                              inout M meta);
control Deparser<H>(packet_out b, in H hdr);

package V1Switch<H, M>(Parser<H, M> p,
                       VerifyChecksum<H, M> vr,
                       Ingress<H, M> ig,
                       Egress<H, M> eg,
                       ComputeChecksum<H, M> ck,
                       Deparser<H> dep
                       );

// CHECK:   p4hir.package @V1Switch<[!type_H, !type_M]>("p" : !Parser_type_H_type_M, "vr" : !VerifyChecksum_type_H_type_M, "ig" : !Ingress_type_H_type_M, "eg" : !Egress_type_H_type_M, "ck" : !ComputeChecksum_type_H_type_M, "dep" : !Deparser_type_H)
control c() {
    apply {
    }
}


control e();
package top(e _e);

// CHECK:  p4hir.package @top("_e" : !e)
// CHECK:  %[[c:.*]] = p4hir.instantiate @c() as "c" : () -> !c
// CHECK:  p4hir.instantiate @top(%[[c]]) as "main" : (!c) -> !top
top(c()) main;
