// RUN: p4mlir-opt --verify-roundtrip %s | FileCheck %s

!packet_in = !p4hir.extern<"packet_in">
!packet_out = !p4hir.extern<"packet_out">
!top = !p4hir.package<"top">
!c = !p4hir.control<"c", ()>
!e = !p4hir.control<"e", ()>
!type_H = !p4hir.type_var<"H">
!type_M = !p4hir.type_var<"M">
!error = !p4hir.error<NoError, PacketTooShort, NoMatch, StackOutOfBounds, HeaderTooShort, ParserTimeout, ParserInvalidArgument>
!Deparser_type_H = !p4hir.control<"Deparser"<!type_H>, (!packet_out, !type_H)>
!standard_metadata_t = !p4hir.struct<"standard_metadata_t">
!ComputeChecksum_type_H_type_M = !p4hir.control<"ComputeChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!VerifyChecksum_type_H_type_M = !p4hir.control<"VerifyChecksum"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>)>
!Egress_type_H_type_M = !p4hir.control<"Egress"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Ingress_type_H_type_M = !p4hir.control<"Ingress"<!type_H, !type_M>, (!p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>
!Parser_type_H_type_M = !p4hir.parser<"Parser"<!type_H, !type_M>, (!packet_in, !p4hir.ref<!type_H>, !p4hir.ref<!type_M>, !p4hir.ref<!standard_metadata_t>)>

module {
  // CHECK: module
  p4hir.control @c()() {
    p4hir.control_apply {
      p4hir.return
    }
  }
  p4hir.package @V1Switch<[!type_H, !type_M]>("p" : !Parser_type_H_type_M, "vr" : !VerifyChecksum_type_H_type_M, "ig" : !Ingress_type_H_type_M, "eg" : !Egress_type_H_type_M, "ck" : !ComputeChecksum_type_H_type_M, "dep" : !Deparser_type_H)
  p4hir.package @top("_e" : !e)
  %c = p4hir.instantiate @c() as "c" : () -> !c
  %main = p4hir.instantiate @top(%c) as "main" : (!c) -> !top
}
