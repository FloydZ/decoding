#![allow(non_snake_case)]
use mceliece::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let (syndrom, H) =
        read_instace(mceliece::instance::s.to_string(), mceliece::instance::h.to_string());

    c.bench_function("prange 240", |b| b.iter(||
        unsafe { prange(black_box(&syndrom), black_box(&H)) }
    ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);