//! Interpreter tests: run real Aurora programs and check their output/result.

use crate::{run, Value};
use aurora_parser::parse_str;

/// Parse and run `main`, returning (result value, captured output).
fn exec(src: &str) -> (Value, String) {
    let (module, diags) = parse_str(src);
    assert!(
        !diags.iter().any(|d| d.is_error()),
        "parse errors: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let (result, output) = run(&module, "main");
    (result.expect("runtime error"), output)
}

fn out(src: &str) -> String {
    exec(src).1
}

#[test]
fn arithmetic_and_println() {
    assert_eq!(out("fn main() { println(1 + 2 * 3) }"), "7\n");
    assert_eq!(out("fn main() { println(10 / 3) }"), "3\n");
    assert_eq!(out("fn main() { println(2.5 + 0.5) }"), "3\n");
}

#[test]
fn let_bindings_and_reassignment() {
    let src = "fn main() {
        let mut x = 1
        x = x + 41
        println(x)
    }";
    assert_eq!(out(src), "42\n");
}

#[test]
fn compound_assignment() {
    let src = "fn main() {
        let mut total = 0
        total += 10
        total *= 4
        total -= 2
        println(total)
    }";
    assert_eq!(out(src), "38\n");
}

#[test]
fn if_else_and_comparison() {
    let src = "fn main() {
        let x = 7
        if x > 5 { println(\"big\") } else { println(\"small\") }
    }";
    assert_eq!(out(src), "big\n");
}

#[test]
fn while_loop_sum() {
    let src = "fn main() {
        let mut i = 1
        let mut sum = 0
        while i <= 5 {
            sum += i
            i += 1
        }
        println(sum)
    }";
    assert_eq!(out(src), "15\n"); // 1+2+3+4+5
}

#[test]
fn for_range_loop() {
    let src = "fn main() {
        let mut acc = 0
        for i in 1..=4 { acc += i }
        println(acc)
    }";
    assert_eq!(out(src), "10\n");
}

#[test]
fn recursion_factorial() {
    let src = "fn fact(n: i32) -> i32 {
        if n <= 1 { return 1 }
        n * fact(n - 1)
    }
    fn main() { println(fact(5)) }";
    assert_eq!(out(src), "120\n");
}

#[test]
fn recursion_fibonacci() {
    let src = "fn fib(n: i32) -> i32 {
        if n < 2 { return n }
        fib(n - 1) + fib(n - 2)
    }
    fn main() { println(fib(10)) }";
    assert_eq!(out(src), "55\n");
}

#[test]
fn structs_and_field_access() {
    let src = "struct Point { x: i32, y: i32 }
    fn main() {
        let mut p = Point { x: 3, y: 4 }
        p.x = p.x + 10
        println(p.x)
        println(p.y)
    }";
    assert_eq!(out(src), "13\n4\n");
}

#[test]
fn arrays_index_and_mutate() {
    let src = "fn main() {
        let mut a = [10, 20, 30]
        a[1] = 99
        println(a[0])
        println(a[1])
    }";
    assert_eq!(out(src), "10\n99\n");
}

#[test]
fn tuple_destructuring() {
    let src = "fn main() {
        let (a, b) = (1, 2)
        println(a + b)
    }";
    assert_eq!(out(src), "3\n");
}

#[test]
fn break_and_continue() {
    let src = "fn main() {
        let mut i = 0
        let mut seen = 0
        while true {
            i += 1
            if i == 3 { continue }
            if i > 5 { break }
            seen += i
        }
        println(seen)
    }"; // 1+2+4+5 = 12 (skips 3, stops after 5)
    assert_eq!(out(src), "12\n");
}

#[test]
fn return_value_from_main() {
    let (v, _) = exec("fn main() -> i32 { 42 }");
    assert_eq!(v, Value::Int(42));
}

#[test]
fn division_by_zero_is_runtime_error() {
    let (module, _) = parse_str("fn main() { println(1 / 0) }");
    let (result, _) = run(&module, "main");
    assert!(result.unwrap_err().contains("division by zero"));
}

#[test]
fn string_concatenation() {
    assert_eq!(out("fn main() { println(\"foo\" + \"bar\") }"), "foobar\n");
}

// --- closures ------------------------------------------------------------

#[test]
fn closure_basic_call() {
    let src = "fn main() {
        let add = |a, b| a + b
        println(add(2, 3))
    }";
    assert_eq!(out(src), "5\n");
}

#[test]
fn closure_captures_environment() {
    let src = "fn main() {
        let base = 100
        let bump = |x| x + base
        println(bump(5))
    }";
    assert_eq!(out(src), "105\n");
}

#[test]
fn higher_order_function() {
    let src = "fn apply(f: fn(i32) -> i32, v: i32) -> i32 { f(v) }
    fn main() {
        println(apply(|n| n * 2, 21))
    }";
    assert_eq!(out(src), "42\n");
}

#[test]
fn closure_with_block_body() {
    let src = "fn main() {
        let classify = |n| {
            if n > 0 { return 1 }
            if n < 0 { return -1 }
            0
        }
        println(classify(7))
        println(classify(-3))
        println(classify(0))
    }";
    assert_eq!(out(src), "1\n-1\n0\n");
}

#[test]
fn pipe_operator_threads_value_as_first_arg() {
    let src = "fn double(x: i32) -> i32 { x * 2 }
    fn add(x: i32, y: i32) -> i32 { x + y }
    fn main() {
        // 5 |> add(10) == add(5, 10) == 15; then |> double == double(15) == 30
        println(5 |> add(10) |> double)
    }";
    assert_eq!(out(src), "30\n");
}

// --- math builtins -------------------------------------------------------

#[test]
fn math_builtins() {
    assert_eq!(out("fn main() { println(sqrt(16.0)) }"), "4\n");
    assert_eq!(out("fn main() { println(abs(-5)) }"), "5\n"); // int abs stays int
    assert_eq!(out("fn main() { println(max(3, 7)) }"), "7\n");
    assert_eq!(out("fn main() { println(min(3.5, 1.5)) }"), "1.5\n");
    assert_eq!(out("fn main() { println(pow(2.0, 10.0)) }"), "1024\n");
    assert_eq!(out("fn main() { println(floor(3.9)) }"), "3\n");
    assert_eq!(out("fn main() { println(clamp(12.0, 0.0, 9.0)) }"), "9\n");
}

#[test]
fn str_conversion_builtin() {
    assert_eq!(out("fn main() { println(str(42)) }"), "42\n");
    assert_eq!(out("fn main() { println(str(true)) }"), "true\n");
    // Build a label by concatenating.
    assert_eq!(out("fn main() { println(\"hp=\" + str(75)) }"), "hp=75\n");
}

#[test]
fn len_builtin() {
    assert_eq!(out("fn main() { println(len([10, 20, 30])) }"), "3\n");
    assert_eq!(out("fn main() { println(len(\"hello\")) }"), "5\n");
    let src = "fn main() {
        let mut total = 0
        let xs = [4, 5, 6, 7]
        for i in 0..len(xs) { total += xs[i] }
        println(total)
    }";
    assert_eq!(out(src), "22\n");
}

#[test]
fn math_builtins_compose_for_geometry() {
    // Euclidean distance via sqrt + arithmetic — the kind of thing gameplay needs.
    let src = "fn dist(x: f64, y: f64) -> f64 { sqrt(x * x + y * y) }
    fn main() { println(dist(3.0, 4.0)) }";
    assert_eq!(out(src), "5\n");
}

// --- builtin graphics ----------------------------------------------------

#[test]
fn aurora_program_draws_a_triangle() {
    // An Aurora program drives the CPU rasterizer via builtins and inspects the
    // resulting pixels with fb_get.
    let src = "fn main() {
        framebuffer(20, 20)
        clear(0, 0, 0)
        triangle(2, 2, 18, 2, 2, 18, 255, 40, 40)
        println(fb_get(4, 4))    // inside -> red, packed 0xFF2828
        println(fb_get(17, 17))  // outside -> background 0
    }";
    assert_eq!(out(src), "16721960\n0\n"); // 0xFF2828 = 16721960
}

// --- enums ---------------------------------------------------------------

#[test]
fn enum_unit_variant_match() {
    let src = "enum Dir { North, South, East, West }
    fn opposite(d: Dir) -> Dir {
        match d {
            Dir::North => Dir::South,
            Dir::South => Dir::North,
            _ => d,
        }
    }
    fn main() {
        println(opposite(Dir::North))
    }";
    assert_eq!(out(src), "Dir::South\n");
}

#[test]
fn enum_tuple_variant_payload() {
    let src = "enum Shape { Circle(i32), Rect(i32, i32) }
    fn area(s: Shape) -> i32 {
        match s {
            Shape::Circle(r) => 3 * r * r,
            Shape::Rect(w, h) => w * h,
        }
    }
    fn main() {
        println(area(Shape::Circle(2)))   // 12
        println(area(Shape::Rect(3, 4)))  // 12
    }";
    assert_eq!(out(src), "12\n12\n");
}

#[test]
fn enum_struct_variant_payload() {
    let src = "enum Event { Spawn { id: i32 }, Tick }
    fn describe(e: Event) -> i32 {
        match e {
            Event::Spawn { id } => id,
            Event::Tick => 0,
        }
    }
    fn main() {
        println(describe(Event::Spawn { id: 99 }))
        println(describe(Event::Tick))
    }";
    assert_eq!(out(src), "99\n0\n");
}

// --- methods (impl blocks) -----------------------------------------------

#[test]
fn method_call_with_self() {
    let src = "struct Counter { n: i32 }
    impl Counter {
        fn get(&self) -> i32 { self.n }
        fn plus(&self, k: i32) -> i32 { self.n + k }
    }
    fn main() {
        let c = Counter { n: 40 }
        println(c.get())
        println(c.plus(2))
    }";
    assert_eq!(out(src), "40\n42\n");
}

#[test]
fn method_on_enum() {
    let src = "enum Light { Red, Green }
    impl Light {
        fn go(&self) -> bool {
            match self {
                Light::Green => true,
                _ => false,
            }
        }
    }
    fn main() {
        println(Light::Green.go())
        println(Light::Red.go())
    }";
    assert_eq!(out(src), "true\nfalse\n");
}

// --- ECS runtime ---------------------------------------------------------

#[test]
fn spawn_and_query_read() {
    let src = "component Position { x: i32 }
    fn main() {
        spawn(Position { x: 10 })
        spawn(Position { x: 32 })
        let mut total = 0
        for p in query<&Position> { total += p.x }
        println(total)
    }";
    assert_eq!(out(src), "42\n");
}

#[test]
fn system_mutates_components_via_mut_query() {
    // The headline: a system writes &mut components and the changes persist.
    let src = "component Position { x: i32 }
    component Velocity { x: i32 }
    system integrate() {
        for (p, v) in query<&mut Position, &Velocity> {
            p.x += v.x
        }
    }
    fn main() {
        spawn(Position { x: 0 }, Velocity { x: 5 })
        spawn(Position { x: 100 }, Velocity { x: -1 })
        run_systems()
        run_systems()
        let mut total = 0
        for p in query<&Position> { total += p.x }
        println(total)   // (0+10) + (100-2) = 108
    }";
    assert_eq!(out(src), "108\n");
}

#[test]
fn query_filters_by_required_components() {
    // Only entities with BOTH Position and Velocity are matched.
    let src = "component Position { x: i32 }
    component Velocity { x: i32 }
    fn main() {
        spawn(Position { x: 1 }, Velocity { x: 1 })
        spawn(Position { x: 1 })            // no Velocity
        let mut n = 0
        for (p, v) in query<&Position, &Velocity> { n += 1 }
        println(n)   // only the first entity matches
    }";
    assert_eq!(out(src), "1\n");
}

#[test]
fn without_filter_excludes() {
    let src = "component A { v: i32 }
    component Frozen { v: i32 }
    fn main() {
        spawn(A { v: 1 })
        spawn(A { v: 1 }, Frozen { v: 0 })
        let mut n = 0
        for a in query<&A, !Frozen> { n += 1 }
        println(n)   // the frozen one is excluded
    }";
    assert_eq!(out(src), "1\n");
}

#[test]
fn despawn_removes_entity() {
    let src = "component A { v: i32 }
    fn main() {
        let e = spawn(A { v: 1 })
        spawn(A { v: 2 })
        despawn(e)
        println(entity_count())
    }";
    assert_eq!(out(src), "1\n");
}

#[test]
fn match_on_literals_with_wildcard() {
    let src = "fn name(n: i32) -> str {
        match n {
            0 => \"zero\",
            1 => \"one\",
            _ => \"many\",
        }
    }
    fn main() {
        println(name(0))
        println(name(1))
        println(name(7))
    }";
    assert_eq!(out(src), "zero\none\nmany\n");
}

#[test]
fn match_with_binding_and_guard() {
    let src = "fn classify(n: i32) -> str {
        match n {
            x if x < 0 => \"negative\",
            0 => \"zero\",
            _ => \"positive\",
        }
    }
    fn main() {
        println(classify(-3))
        println(classify(0))
        println(classify(9))
    }";
    assert_eq!(out(src), "negative\nzero\npositive\n");
}

#[test]
fn match_on_tuple() {
    let src = "fn main() {
        let p = (1, 2)
        let s = match p {
            (0, 0) => \"origin\",
            (x, y) => \"point\",
        }
        println(s)
    }";
    assert_eq!(out(src), "point\n");
}

#[test]
fn entity_term_binds_id() {
    let src = "component A { v: i32 }
    fn main() {
        spawn(A { v: 7 })
        for (e, a) in query<Entity, &A> { println(a.v) }
    }";
    assert_eq!(out(src), "7\n");
}
