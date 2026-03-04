// RUN: p4mlir-opt --ensure-standard-metadata %s | FileCheck %s

// CHECK:   bmv2ir.header_instance @standard_metadata : !bmv2ir.header<"standard_metadata", [ingress_port:!p4hir.bit<9>, egress_spec:!p4hir.bit<9>, egress_port:!p4hir.bit<9>, instance_type:!p4hir.bit<32>, packet_length:!p4hir.bit<32>, enq_timestamp:!p4hir.bit<32>, enq_qdepth:!p4hir.bit<19>, deq_timedelta:!p4hir.bit<32>, deq_qdepth:!p4hir.bit<19>, ingress_global_timestamp:!p4hir.bit<48>, egress_global_timestamp:!p4hir.bit<48>, mcast_grp:!p4hir.bit<16>, egress_rid:!p4hir.bit<16>, checksum_error:!p4hir.bit<1>, priority:!p4hir.bit<3>, _padding:!p4hir.bit<3>], max_length = 41> {metadata = true}

module {}
