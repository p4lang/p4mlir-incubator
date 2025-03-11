// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

extern Crc16 <T> {
    void hash<U>(in U input_data);
    U id<U>(in U x);
}

// CHECK:  p4hir.extern @Crc16<[!type_T]> {
// CHECK:    p4hir.func @hash<!type_U>(!type_U {p4hir.dir = #in})
// CHECK:    p4hir.func @id<!type_U>(!type_U {p4hir.dir = #in}) -> !type_U
// CHECK:  }

extern ext<H> {
    ext(H v);
    void method<T>(H h, T t);
}

// CHECK:  p4hir.extern @ext<[!type_H]> {
// CHECK:    p4hir.func @ext(!type_H {p4hir.dir = #undir})
// CHECK:    p4hir.func @method<!type_T>(!type_H {p4hir.dir = #undir}, !type_T {p4hir.dir = #undir})
// CHECK:  }

extern ext2<H, V> {
    ext2(H v);
    V method<T>(in H h, in T t);
    H method<T>(in T t);
}

// CHECK:  p4hir.extern @ext2<[!type_H, !type_V]> {
// CHECK:    p4hir.func @ext2(!type_H {p4hir.dir = #undir})
// CHECK:    p4hir.overload_set @method {
// CHECK:      p4hir.func @method_0<!type_T>(!type_H {p4hir.dir = #in}, !type_T {p4hir.dir = #in}) -> !type_V
// CHECK:      p4hir.func @method_1<!type_T>(!type_T {p4hir.dir = #in}) -> !type_H
// CHECK:    }
// CHECK:  }
  
extern X<T> {
  X(T t);
  T method(T t);
}

// CHECK:  p4hir.extern @X<[!type_T]> {
// CHECK:    p4hir.func @X(!type_T {p4hir.dir = #undir})
// CHECK:    p4hir.func @method(!type_T {p4hir.dir = #undir}) -> !type_T
// CHECK:  }

extern Y    {
  Y();
  void method<T>(T t);
}

// CHECK:  p4hir.extern @Y {
// CHECK:    p4hir.func @Y()
// CHECK:    p4hir.func @method<!type_T>(!type_T {p4hir.dir = #undir})
// CHECK:  }
  
extern MyCounter<I> {
    MyCounter(bit<32> size);
    void count(in I index);
}

typedef bit<10> my_counter_index_t;
typedef MyCounter<my_counter_index_t> my_counter_t;

// CHECK:  p4hir.extern @MyCounter<[!type_I]> {
// CHECK:    p4hir.func @MyCounter(!b32i {p4hir.dir = #undir})
// CHECK:    p4hir.func @count(!type_I {p4hir.dir = #in})
// CHECK:  }

// CHECK-LABEL: p4hir.parser @p
parser p() {
    // CHECK:    %[[x:.*]] = p4hir.instantiate @X(%{{.*}}) as "x" : (!i32i) -> !X_i32i
    X<int<32>>(32s0) x;

    // CHECK: %[[y:.*]] = p4hir.instantiate @Y() as "y" : () -> !Y
    Y()          y;


    // CHECK: %[[ex:.*]] = p4hir.instantiate @ext(%{{.*}}) as "ex" : (!b16i) -> !ext_b16i
    ext<bit<16>>(16w0) ex;

    // CHECK: %[[ey:.*]] = p4hir.instantiate @ext2(%{{.*}}) as "ey" : (!b16i) -> !ext2_b16i_void
    ext2<bit<16>, void>(16w0) ey;

    state start {
      // CHECK: p4hir.call_method @X::@method (%[[x]], %{{.*}}) : !X_i32i, (!i32i) -> !i32i
      x.method(0);

      // CHECK: p4hir.call_method @Y::@method<[!b8i]> (%[[y]], %{{.*}}) : !Y, (!b8i) -> ()
      y.method(8w0);

      // CHECK: p4hir.call_method @ext::@method<[!b8i]> (%[[ex]], %{{.*}}, %{{.*}}) : !ext_b16i, (!b16i, !b8i) -> ()
      ex.method(0, 8w0);

      // CHECK: p4hir.call_method @ext2::@method<[!b12i]> (%[[ey]], %{{.*}}) : !ext2_b16i_void, (!b12i) -> !b16i
      // CHECK: p4hir.call_method @ext2::@method<[!b8i]> (%[[ey]], %{{.*}}, %{{.*}}) : !ext2_b16i_void, (!b16i, !b8i) -> ()
      ey.method(ey.method(12w1), 8w0);

      transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @Inner(%arg0: !MyCounter_b10i)()
parser Inner(my_counter_t counter_set) {
    state start {
      // CHECK: p4hir.call_method @MyCounter::@count (%arg0, %{{.*}}) : !MyCounter_b10i, (!b10i) -> ()
      counter_set.count(10w42);
      transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @Test()()
parser Test() {
    // CHECK:  %[[counter_set:.*]] = p4hir.instantiate @MyCounter(%{{.*}}) as "counter_set" : (!b32i) -> !MyCounter_b10i
    my_counter_t(1024) counter_set;
    // CHECK: %[[inner:.*]] = p4hir.instantiate @Inner() as "inner" : () -> !Inner
    Inner() inner;

    state start {
        // CHECK: p4hir.apply %[[inner]](%[[counter_set]]) : !Inner
        inner.apply(counter_set);
        transition accept;
    }
}

