// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

extern Crc16 <T> {
    void hash<U>(in U input_data);
    U id<U>(in U x);
}

// CHECK-LABEL  p4hir.extern @Crc16<[!type_T]> {
// CHECK:    p4hir.func @hash<!type_U>(!type_U {p4hir.dir = #in, p4hir.param_name = "input_data"})
// CHECK:    p4hir.func @id<!type_U>(!type_U {p4hir.dir = #in, p4hir.param_name = "x"}) -> !type_U
// CHECK:  }

extern ext<H> {
    ext(H v);
    void method<T>(H h, T t);
}

// CHECK-LABEL:  p4hir.extern @ext<[!type_H]> {
// CHECK:    p4hir.func @ext(!type_H {p4hir.dir = #undir, p4hir.param_name = "v"})
// CHECK:    p4hir.func @method<!type_T>(!type_H {p4hir.dir = #undir, p4hir.param_name = "h"}, !type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
// CHECK:  }

extern ext2<H, V> {
    ext2(H v);
    V method<T>(in H h, in T t);
    H method<T>(in T t);
}

// CHECK-LABEL:  p4hir.extern @ext2<[!type_H, !type_V]> {
// CHECK:    p4hir.func @ext2(!type_H {p4hir.dir = #undir, p4hir.param_name = "v"})
// CHECK:    p4hir.overload_set @method {
// CHECK:      p4hir.func @method_0<!type_T>(!type_H {p4hir.dir = #in, p4hir.param_name = "h"}, !type_T {p4hir.dir = #in, p4hir.param_name = "t"}) -> !type_V
// CHECK:      p4hir.func @method_1<!type_T>(!type_T {p4hir.dir = #in, p4hir.param_name = "t"}) -> !type_H
// CHECK:    }
// CHECK:  }
  
extern X<T> {
  X(T t);
  T method(T t);
}

// CHECK-LABEL:  p4hir.extern @X<[!type_T]> {
// CHECK:    p4hir.func @X(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
// CHECK:    p4hir.func @method(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"}) -> !type_T
// CHECK:  }

extern Y    {
  Y();
  void method<T>(T t);
}

// CHECK-LABEL:  p4hir.extern @Y {
// CHECK:    p4hir.func @Y()
// CHECK:    p4hir.func @method<!type_T>(!type_T {p4hir.dir = #undir, p4hir.param_name = "t"})
// CHECK:  }
  
extern MyCounter<I> {
    MyCounter(bit<32> size);
    void count(in I index);
}

typedef bit<10> my_counter_index_t;
typedef MyCounter<my_counter_index_t> my_counter_t;

// CHECK-LABEL:  p4hir.extern @MyCounter<[!type_I]> {
// CHECK:    p4hir.func @MyCounter(!b32i {p4hir.dir = #undir, p4hir.param_name = "size"})
// CHECK:    p4hir.func @count(!type_I {p4hir.dir = #in, p4hir.param_name = "index"})
// CHECK:  }

// CHECK-LABEL: p4hir.parser @p
parser p() {
    // CHECK:    p4hir.instantiate @X<[!i32i]> (%{{.*}} : !i32i) as @x
    X<int<32>>(32s0) x;

    // CHECK:    p4hir.instantiate @Y () as @y
    Y()          y;


    // CHECK: p4hir.instantiate @ext<[!b16i]> (%{{.*}} : !b16i) as @ex
    ext<bit<16>>(16w0) ex;

    // CHECK: p4hir.instantiate @ext2<[!b16i, !void]> (%{{.*}}) as @ey
    ext2<bit<16>, void>(16w0) ey;

    state start {
      // CHECK: p4hir.call_method @x::@method(%{{.*}}) : (!i32i) -> !i32i
      x.method(0);

      // CHECK: p4hir.call_method @y::@method<[!b8i]>(%{{.*}}) : (!b8i) -> ()
      y.method(8w0);

      // CHECK: p4hir.call_method @ex::@method<[!b8i]>(%{{.*}}, %{{.*}}) : (!b16i, !b8i) -> ()
      ex.method(0, 8w0);

      // CHECK: p4hir.call_method @ey::@method<[!b12i]>(%{{.*}}) : (!b12i) -> !b16i
      // CHECK: p4hir.call_method @ey::@method<[!b8i]>(%{{.*}}, %{{.*}}) : (!b16i, !b8i) -> ()
      ey.method(ey.method(12w1), 8w0);

      transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @Inner
parser Inner(my_counter_t counter_set) {
    state start {
      // CHECK: p4hir.call_method @MyCounter::@count of %arg0 : !MyCounter_b10i (%{{.*}}) : (!b10i) -> ()
      counter_set.count(10w42);
      transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @Test()()
parser Test() {
    // CHECK:  p4hir.instantiate @MyCounter<[!b10i]> (%{{.*}} : !b32i) as @counter_set
    my_counter_t(1024) counter_set;
    // CHECK: p4hir.instantiate @Inner () as @inner
    Inner() inner;

    state start {
        // xHECK: p4hir.apply %[[inner]](%[[counter_set]]) : !Inner
        inner.apply(counter_set);
        transition accept;
    }
}

control Inner2(my_counter_t counter_set) {
    apply {
      counter_set.count(10w42);
    }
}
control Test2() {
    my_counter_t(42) counter_set;
    // CHECK: p4hir.instantiate @Inner () as @inner
    Inner() inner;

    apply {
        inner.apply(counter_set);
    }
}
