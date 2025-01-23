// RUN: p4mlir-opt %s | FileCheck %s

!s8i = !p4hir.int<8>
!s16i = !p4hir.int<16>
!s32i = !p4hir.int<32>
!s64i = !p4hir.int<64>

!u8i = !p4hir.bit<8>
!u16i = !p4hir.bit<16>
!u32i = !p4hir.bit<32>
!u64i = !p4hir.bit<64>

// No need to check stuff. If it parses, it's fine.
// CHECK: module
module {
    %1 = p4hir.const #p4hir.int<-128> : !p4hir.int<8>
    %2 = p4hir.const #p4hir.int<127> : !p4hir.int<8>
    %3 = p4hir.const #p4hir.int<255> : !p4hir.bit<8>

    %4 = p4hir.const #p4hir.int<-32768> : !p4hir.int<16>
    %5 = p4hir.const #p4hir.int<32767> : !p4hir.int<16>
    %6 = p4hir.const #p4hir.int<65535> : !p4hir.bit<16>

    %7 = p4hir.const #p4hir.int<-2147483648> : !p4hir.int<32>
    %8 = p4hir.const #p4hir.int<2147483647> : !p4hir.int<32>
    %9 = p4hir.const #p4hir.int<4294967295> : !p4hir.bit<32>

    %10 = p4hir.const #p4hir.int<9223372036854775807> : !p4hir.int<72>
    %11 = p4hir.const #p4hir.int<18446744073709551615> : !p4hir.bit<72>
    %13 = p4hir.const #p4hir.int<-9223372036854775807> : !p4hir.int<72>

    %12 = p4hir.const #p4hir.int<5> : !p4hir.infint
    %14 = p4hir.const #p4hir.int<-100500> : !p4hir.infint
}
