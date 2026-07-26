//! Persistence cost of cumulative `BashOutputChunk` snapshots.
//!
//! The terminal actor emits a cumulative snapshot every roughly 100 ms. The
//! notification bridge turns each snapshot into an in-progress
//! `ToolCallUpdate`, which `SessionPersistence` currently writes through the
//! production JSONL adapter. This benchmark sends that exact ACP/`BashOutput`
//! representation through the real persistence actor; fixture creation is
//! outside Criterion timing, while a measured operation sends one complete
//! tool-call sequence at the terminal's 100 ms virtual cadence and waits for
//! its durable flush barrier.
//!
//! The workload is 16 snapshots of a command that has produced 64 KiB total
//! ASCII output over 1.6 s. It mirrors the default 20,000-character terminal
//! limit: each progress snapshot is the retained tail after
//! `ProcessState::maybe_truncate`, while the independent terminal update has
//! the source's front-and-tail result. The audit lines report the actual JSONL
//! bytes and records added by one complete command.
//!
//! Run: `cargo bench -p xai-grok-shell --bench bash_output_persistence`

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use agent_client_protocol as acp;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use tempfile::TempDir;
use xai_grok_shell::session::info::Info;
use xai_grok_shell::session::persistence::{
    PersistenceHandle, PersistenceMsg, new_with_explicit_dir,
};
use xai_grok_shell::session::storage::SessionUpdate;
use xai_grok_tools::DEFAULT_TOOL_OUTPUT_CHARS;
use xai_grok_tools::types::output::BashOutput;

const CHUNK_COUNT: usize = 16;
const BYTES_PER_CHUNK: usize = 4 * 1024;
const CHUNK_INTERVAL: Duration = Duration::from_millis(100);

struct Fixture {
    _root: TempDir,
    persistence: PersistenceHandle,
    updates_path: PathBuf,
}

fn notification(
    session_id: &acp::SessionId,
    event_id: impl Into<String>,
    update: acp::SessionUpdate,
) -> SessionUpdate {
    let mut meta = serde_json::Map::new();
    meta.insert(
        "eventId".to_owned(),
        serde_json::Value::String(event_id.into()),
    );
    SessionUpdate::Acp(Box::new(
        acp::SessionNotification::new(session_id.clone(), update).meta(Some(meta)),
    ))
}

/// Mirror `ProcessState::maybe_truncate` for this ASCII fixture. The source
/// counts characters, so byte and character offsets intentionally coincide here.
fn truncate_default_tail(output: &mut Vec<u8>, front: &mut Option<Vec<u8>>) -> bool {
    let rendered = String::from_utf8_lossy(output);
    let char_count = rendered.chars().count();
    if char_count <= DEFAULT_TOOL_OUTPUT_CHARS {
        return false;
    }

    let half = DEFAULT_TOOL_OUTPUT_CHARS / 2;
    if front.is_none() {
        let front_end = rendered
            .char_indices()
            .nth(half)
            .map(|(index, _)| index)
            .unwrap_or(rendered.len());
        *front = Some(rendered[..front_end].as_bytes().to_vec());
    }
    let tail_start_char = char_count.saturating_sub(half);
    let tail_start_byte = rendered
        .char_indices()
        .nth(tail_start_char)
        .map(|(index, _)| index)
        .unwrap_or(rendered.len());
    let tail = rendered[tail_start_byte..].as_bytes().to_vec();
    drop(rendered);
    *output = tail;
    true
}

fn cumulative_bash_updates(session_id: &acp::SessionId) -> Vec<SessionUpdate> {
    let tool_call_id = acp::ToolCallId::new("bash-output-bench");
    let mut output_tail = Vec::with_capacity(DEFAULT_TOOL_OUTPUT_CHARS);
    let mut output_front = None;
    let mut total_bytes = 0;
    let mut truncated = false;
    let mut updates = Vec::with_capacity(CHUNK_COUNT + 2);

    // This mirrors the status-less tool-call update sent before a terminal
    // command starts. It must remain ordered ahead of all progress snapshots.
    updates.push(notification(
        session_id,
        "bench-start",
        acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
            tool_call_id.clone(),
            acp::ToolCallUpdateFields::new().title(Some("Run terminal command".to_owned())),
        )),
    ));

    for chunk_index in 1..=CHUNK_COUNT {
        let mut chunk = vec![b'a' + (chunk_index % 26) as u8; BYTES_PER_CHUNK];
        let prefix = format!("chunk-{chunk_index:02}: ");
        chunk[..prefix.len()].copy_from_slice(prefix.as_bytes());
        chunk[BYTES_PER_CHUNK - 1] = b'\n';
        total_bytes += chunk.len();
        output_tail.extend_from_slice(&chunk);
        truncated |= truncate_default_tail(&mut output_tail, &mut output_front);

        // `poll_process` sends the post-truncation output_buffer, not the
        // front-and-tail terminal result.
        let output_text = String::from_utf8_lossy(&output_tail).into_owned();
        let bash_output = BashOutput {
            output: output_tail.clone(),
            output_for_prompt: BashOutput::make_output_for_prompt(&output_text),
            exit_code: 0,
            command: "emit-default-truncated-output".to_owned(),
            truncated,
            signal: None,
            timed_out: false,
            description: None,
            current_dir: "/bench".to_owned(),
            output_file: String::new(),
            total_bytes,
            output_delta: None,
            was_bare_echo: false,
        };
        let update = acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
            tool_call_id.clone(),
            acp::ToolCallUpdateFields::new()
                .status(Some(acp::ToolCallStatus::InProgress))
                .content(Some(vec![acp::ToolCallContent::from(
                    acp::ContentBlock::Text(acp::TextContent::new(output_text)),
                )]))
                .raw_output(serde_json::to_value(&bash_output).expect("BashOutput serializes")),
        ));
        updates.push(notification(
            session_id,
            format!("bench-{chunk_index}"),
            update,
        ));
    }

    // Terminal output is independently emitted by the completed tool-call
    // path. `ProcessState::to_result` rejoins the frozen front and tail.
    let final_output = match output_front {
        Some(front) => format!(
            "{}\n\n... (output truncated) ...\n\n{}",
            String::from_utf8_lossy(&front).trim_end(),
            String::from_utf8_lossy(&output_tail).trim_start()
        ),
        None => String::from_utf8_lossy(&output_tail).into_owned(),
    };
    updates.push(notification(
        session_id,
        "bench-terminal",
        acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
            tool_call_id,
            acp::ToolCallUpdateFields::new()
                .status(Some(acp::ToolCallStatus::Completed))
                .content(Some(vec![acp::ToolCallContent::from(
                    acp::ContentBlock::Text(acp::TextContent::new(final_output)),
                )])),
        )),
    ));

    updates
}

fn fixture(runtime: &tokio::runtime::Runtime) -> Fixture {
    let root = tempfile::tempdir().expect("create benchmark root");
    let session_dir = root.path().join("session");
    let info = Info {
        id: acp::SessionId::new("bash-output-bench"),
        cwd: "/bench".to_owned(),
    };
    let sampling_client =
        xai_grok_shell::sampling::Client::new(xai_grok_sampler::SamplerConfig::default())
            .expect("create persistence sampling client");
    let persistence = runtime
        .block_on(new_with_explicit_dir(
            &info,
            session_dir.clone(),
            acp::ModelId::new("bench-model"),
            sampling_client,
            "bench-model".to_owned(),
        ))
        .expect("start production persistence actor");
    Fixture {
        _root: root,
        persistence,
        updates_path: session_dir.join("updates.jsonl"),
    }
}

fn file_len(path: &std::path::Path) -> u64 {
    fs::metadata(path).map_or(0, |metadata| metadata.len())
}

fn persist_sequence(
    runtime: &tokio::runtime::Runtime,
    fixture: &Fixture,
    updates: &[SessionUpdate],
) -> u64 {
    let before = file_len(&fixture.updates_path);
    runtime.block_on(async {
        for (index, update) in updates.iter().enumerate() {
            fixture
                .persistence
                .tx
                .send(PersistenceMsg::Update(update.clone()))
                .expect("persistence actor remains available");
            // Give the actor a deterministic turn for each terminal tick rather
            // than measuring an all-at-once channel backlog.
            tokio::task::yield_now().await;
            if (1..=CHUNK_COUNT).contains(&index) {
                tokio::time::advance(CHUNK_INTERVAL).await;
                tokio::task::yield_now().await;
            }
        }
        let (respond_to, acknowledged) = tokio::sync::oneshot::channel();
        fixture
            .persistence
            .tx
            .send(PersistenceMsg::FlushAndAck { respond_to })
            .expect("queue durable flush barrier");
        acknowledged
            .await
            .expect("persistence actor acknowledges durable flush");
    });
    file_len(&fixture.updates_path) - before
}

fn audit(runtime: &tokio::runtime::Runtime, updates: &[SessionUpdate]) -> (u64, u64) {
    let fixture = fixture(runtime);
    let bytes = persist_sequence(runtime, &fixture, updates);
    let contents = fs::read_to_string(&fixture.updates_path).expect("read persisted updates");
    let records = contents.lines().count() as u64;

    assert!(records > 0, "a completed tool call must remain replayable");
    assert!(bytes > 0, "the durable flush must add JSONL content");
    for line in contents.lines() {
        let _: serde_json::Value = serde_json::from_str(line).expect("JSONL line remains valid");
    }

    (bytes, records)
}

fn bench_bash_output_persistence(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("create benchmark runtime");
    runtime.block_on(async { tokio::time::pause() });
    let audit_updates = cumulative_bash_updates(&acp::SessionId::new("bash-output-bench"));
    let (persisted_bytes, persisted_records) = audit(&runtime, &audit_updates);
    println!("PERFLOOP_METRIC\tpersisted_bytes/op\t{persisted_bytes}");
    println!("PERFLOOP_METRIC\tpersisted_records/op\t{persisted_records}");

    let measured_fixture = fixture(&runtime);
    let measured_updates = cumulative_bash_updates(&acp::SessionId::new("bash-output-bench"));
    let mut group = c.benchmark_group("bash_output_persistence");
    group.throughput(Throughput::Elements(1));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(200));
    group.bench_function("16_default_truncated_chunks_64KiB", |b| {
        b.iter_batched(
            || (),
            |_| {
                black_box(persist_sequence(
                    &runtime,
                    &measured_fixture,
                    black_box(&measured_updates),
                ))
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_bash_output_persistence);
criterion_main!(benches);
