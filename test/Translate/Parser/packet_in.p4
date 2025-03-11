// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

extern packet_in {
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader,
                    in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(in bit<32> sizeInBits);
    bit<32> length();
}

// CHECK:  ![[packet_in:.*]] = !p4hir.extern<"packet_in">
// CHECK:  ![[type_T:.*]] = !p4hir.type_var<"T">
// CHECK:  p4hir.extern @packet_in {
// CHECK:    p4hir.overload_set @extract {
// CHECK:      p4hir.func @extract_0<![[type_T]]>(!p4hir.ref<![[type_T]]> {p4hir.dir = #out})
// CHECK:      p4hir.func @extract_1<![[type_T]]>(!p4hir.ref<![[type_T]]> {p4hir.dir = #out}, !b32i {p4hir.dir = #in})
// CHECK:    }
// CHECK:    p4hir.func @lookahead<![[type_T]]>() -> ![[type_T]]
// CHECK:    p4hir.func @advance(!b32i {p4hir.dir = #in})
// CHECK:    p4hir.func @length() -> !b32i
// CHECK:  }

typedef bit<48> EthernetAddress;

header Ethernet_h {
    EthernetAddress dstAddr;
    EthernetAddress srcAddr;
    bit<16>         etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<6>  dscp;
    bit<2>  ecn;
    bit<16> totalLen;
    bit<16> identification;
    bit<1>  flag_rsvd;
    bit<1>  flag_noFrag;
    bit<1>  flag_more;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

struct Parsed_packet {
    Ethernet_h    ethernet;
    ipv4_t        ipv4;
}

parser parserI(packet_in pkt,
               out Parsed_packet hdr) {
    state start {
// CHECK: p4hir.call_method @packet_in::@extract<[!Ethernet_h]> (%{{.*}}, %{{.*}}) : ![[packet_in]], (!p4hir.ref<!Ethernet_h>) -> ()
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
          16w0x0800: parse_ipv4;
          default: accept;
        }
    }
    state parse_ipv4 {
// CHECK: p4hir.call_method @packet_in::@extract<[!ipv4_t]> (%{{.*}}, %{{.*}}) : ![[packet_in]], (!p4hir.ref<!ipv4_t>) -> ()    
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.version, hdr.ipv4.protocol) {
          (4w0x4, 8w0x06): accept;
          (4w0x4, 8w0x17): accept;
          default: accept;
        }
    }
}
