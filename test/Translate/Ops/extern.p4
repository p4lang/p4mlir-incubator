// RUN: p4mlir-translate --typeinference-only %s | FileCheck %s

extern Crc16 <T> {
    void hash<U>(in U input_data);
    U id<U>(in U x);
}

// CHECK:  p4hir.extern @Crc16<[!p4hir.type_var<"T">]> {
// CHECK:    p4hir.func @hash<!p4hir.type_var<"U">>(!p4hir.type_var<"U"> {p4hir.dir = #in})
// CHECK:    p4hir.func @id<!p4hir.type_var<"U">>(!p4hir.type_var<"U"> {p4hir.dir = #in}) -> !p4hir.type_var<"U">
// CHECK:  }

extern ext<H> {
    ext(H v);
    void method<T>(H h, T t);
}

// CHECK:  p4hir.extern @ext<[!p4hir.type_var<"H">]> {
// CHECK:    p4hir.func @ext(!p4hir.type_var<"H"> {p4hir.dir = #undir})
// CHECK:    p4hir.func @method<!p4hir.type_var<"T">>(!p4hir.type_var<"H"> {p4hir.dir = #undir}, !p4hir.type_var<"T"> {p4hir.dir = #undir})
// CHECK:  }

extern ext2<H, V> {
    ext2(H v);
    V method<T>(in H h, in T t);
    H method<T>(in T t);
}

// CHECK:  p4hir.extern @ext2<[!p4hir.type_var<"H">, !p4hir.type_var<"V">]> {
// CHECK:    p4hir.func @ext2(!p4hir.type_var<"H"> {p4hir.dir = #undir})
// CHECK:    p4hir.overload_set @method {
// CHECK:      p4hir.func @method_0<!p4hir.type_var<"T">>(!p4hir.type_var<"H"> {p4hir.dir = #in}, !p4hir.type_var<"T"> {p4hir.dir = #in}) -> !p4hir.type_var<"V">
// CHECK:      p4hir.func @method_1<!p4hir.type_var<"T">>(!p4hir.type_var<"T"> {p4hir.dir = #in}) -> !p4hir.type_var<"H">
// CHECK:    }
// CHECK:  }
  
extern X<T> {
  X(T t);
  T method(T t);
}

// CHECK:  p4hir.extern @X<[!p4hir.type_var<"T">]> {
// CHECK:    p4hir.func @X(!p4hir.type_var<"T"> {p4hir.dir = #undir})
// CHECK:    p4hir.func @method(!p4hir.type_var<"T"> {p4hir.dir = #undir}) -> !p4hir.type_var<"T">
// CHECK:  }

extern Y    {
  Y();
  void method<T>(T t);
}

// CHECK:  p4hir.extern @Y {
// CHECK:    p4hir.func @Y()
// CHECK:    p4hir.func @method<!p4hir.type_var<"T">>(!p4hir.type_var<"T"> {p4hir.dir = #undir})
// CHECK:  }
  
extern MyCounter<I> {
    MyCounter(bit<32> size);
    void count(in I index);
}

typedef bit<10> my_counter_index_t;
typedef MyCounter<my_counter_index_t> my_counter_t;

// CHECK:  p4hir.extern @MyCounter<[!p4hir.type_var<"I">]> {
// CHECK:    p4hir.func @MyCounter(!b32i {p4hir.dir = #undir})
// CHECK:    p4hir.func @count(!p4hir.type_var<"I"> {p4hir.dir = #in})
// CHECK:  }

// CHECK-LABEL: p4hir.parser @p
parser p() {
    // CHECK:    %[[x:.*]] = p4hir.instantiate @X(%{{.*}}) as "x" : (!i32i) -> !p4hir.extern<"X"<!i32i>>
    X<int<32>>(32s0) x;

    // CHECK: %[[y:.*]] = p4hir.instantiate @Y() as "y" : () -> !p4hir.extern<"Y">
    Y()          y;


    // CHECK: %[[ex:.*]] = p4hir.instantiate @ext(%{{.*}}) as "ex" : (!b16i) -> !p4hir.extern<"ext"<!b16i>>
    ext<bit<16>>(16w0) ex;

    // CHECK: %[[ey:.*]] = p4hir.instantiate @ext2(%{{.*}}) as "ey" : (!b16i) -> !p4hir.extern<"ext2"<!b16i, !void>>
    ext2<bit<16>, void>(16w0) ey;

    state start {
      // CHECK: p4hir.call_method @X::@method (%[[x]], %{{.*}}) : !p4hir.extern<"X"<!i32i>>, (!i32i) -> !i32i
      x.method(0);

      // CHECK: p4hir.call_method @Y::@method<[!b8i]> (%[[y]], %{{.*}}) : !p4hir.extern<"Y">, (!b8i) -> ()
      y.method(8w0);

      // CHECK: p4hir.call_method @ext::@method<[!b8i]> (%[[ex]], %{{.*}}, %{{.*}}) : !p4hir.extern<"ext"<!b16i>>, (!b16i, !b8i) -> ()
      ex.method(0, 8w0);

      // CHECK: p4hir.call_method @ext2::@method<[!b12i]> (%[[ey]], %{{.*}}) : !p4hir.extern<"ext2"<!b16i, !void>>, (!b12i) -> !b16i
      // CHECK: p4hir.call_method @ext2::@method<[!b8i]> (%[[ey]], %{{.*}}, %{{.*}}) : !p4hir.extern<"ext2"<!b16i, !void>>, (!b16i, !b8i) -> ()
      ey.method(ey.method(12w1), 8w0);

      transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @Inner(%arg0: !p4hir.extern<"MyCounter"<!b10i>>)()
parser Inner(my_counter_t counter_set) {
    state start {
      // CHECK: p4hir.call_method @MyCounter::@count (%arg0, %{{.*}}) : !p4hir.extern<"MyCounter"<!b10i>>, (!b10i) -> ()
      counter_set.count(10w42);
      transition accept;
    }
}

// CHECK-LABEL: p4hir.parser @Test()()
parser Test() {
    // CHECK:  %[[counter_set:.*]] = p4hir.instantiate @MyCounter(%{{.*}}) as "counter_set" : (!b32i) -> !p4hir.extern<"MyCounter"<!b10i>>
    my_counter_t(1024) counter_set;
    // CHECK: %[[inner:.*]] = p4hir.instantiate @Inner() as "inner" : () -> !p4hir.parser<"Inner", (!p4hir.extern<"MyCounter"<!b10i>>)>
    Inner() inner;

    state start {
        // CHECK: p4hir.apply %[[inner]](%[[counter_set]]) : !p4hir.parser<"Inner", (!p4hir.extern<"MyCounter"<!b10i>>)>
        inner.apply(counter_set);
        transition accept;
    }
}

