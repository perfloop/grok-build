//! Criterion measurement for the consecutive streamed ACP text merge path.
//!
//! The fixture builds 128 runtime-generated 1 KiB `AgentMessageChunk`s before
//! measurement. Each iteration mutates the actor-owned pending notification and
//! validates the final concatenated text, matching the persistence actor's
//! repeated merge path without storage or channel work.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use xai_grok_shell::session::persistence::persistence_merge_fixture::MergeStreamFixture;

#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

fn allocation_metrics() -> (u64, u64) {
    // Construct the fixture before profiling so payload generation and actor
    // setup do not contribute to the allocation mechanism measurement.
    let mut fixture = MergeStreamFixture::new();
    let profiler = dhat::Profiler::builder().testing().build();
    let before = dhat::HeapStats::get();
    black_box(fixture.merge_stream());
    let after = dhat::HeapStats::get();
    fixture.clear_pending();
    drop(profiler);

    (
        after.total_bytes - before.total_bytes,
        after.total_blocks - before.total_blocks,
    )
}

fn bench_persistence_merge(criterion: &mut Criterion) {
    let (allocated_bytes, allocation_blocks) = allocation_metrics();
    println!(
        "PERFLOOP_JSONL {}",
        serde_json::json!({
            "metric": "allocated_bytes_per_stream",
            "value": allocated_bytes,
        })
    );
    println!(
        "PERFLOOP_JSONL {}",
        serde_json::json!({
            "metric": "allocation_blocks_per_stream",
            "value": allocation_blocks,
        })
    );

    let mut group = criterion.benchmark_group("persistence_merge");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(200))
        .sampling_mode(SamplingMode::Flat);
    group.bench_function("128x1024_text_stream", |bencher| {
        let mut fixture = MergeStreamFixture::new();
        bencher.iter(|| black_box(fixture.merge_stream()));
    });
    group.finish();
}

criterion_group!(benches, bench_persistence_merge);
criterion_main!(benches);
