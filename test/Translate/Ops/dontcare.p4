// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

struct intrinsic_metadata_t {
    bit<8> f0;
    bit<8> f1;
 }

 struct empty_t {}

header Header {
    bit<32> data;
}

// CHECK: ![[P:.*]] = !p4hir.package<"P"<!p4hir.dontcare, !p4hir.dontcare>>

extern packet_in {
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader,
                    in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(in bit<32> sizeInBits);
    bit<32> length();
}

// CHECK-LABEL: p4hir.parser @p0
// CHECK-LABEL: p4hir.state @start
parser p0(packet_in p, out Header h) {
    state start {
    // CHECK: p4hir.variable ["dummy"] : <!Header>
        p.extract<Header>(_);
        transition next;
    }

    state next {
        p.extract(h);
        transition accept;
    }
}

 control nothing(inout empty_t hdr, inout empty_t meta, in intrinsic_metadata_t imeta) {
    apply {}
 }

 control C<H, M>(
     inout H hdr,
     inout M meta,
     in intrinsic_metadata_t intr_md);

 package P<H, M>(C<H, M> c = nothing());

 struct hdr_t { }
 struct meta_t { }

// CHECK: p4hir.instantiate @P(%nothing) as "main" : (!nothing) -> ![[P]]
 P<_, _>() main;
