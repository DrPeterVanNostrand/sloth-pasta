use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn pallas(c: &mut Criterion) {
    let one = pasta_curves::Fp::one();
    let one_asm = sloth_pasta::Fp::one();

    let mut group = c.benchmark_group("pallas-add");
    group.bench_function("non-asm", |b| b.iter(|| black_box(one.add(&one))));
    group.bench_function("asm", |b| b.iter(|| black_box(one_asm.add(&one_asm))));
    group.finish();

    let mut group = c.benchmark_group("pallas-mul");
    group.bench_function("non-asm", |b| b.iter(|| black_box(one.add(&one))));
    group.bench_function("asm", |b| b.iter(|| black_box(one_asm.mul(&one_asm))));
    group.finish();
}

criterion_group!(benches, pallas);
criterion_main!(benches);
