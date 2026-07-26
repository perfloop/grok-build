//! Focused end-to-end regression and measurement coverage for incremental
//! session-search maintenance.
//!
//! The production path is `notify_session_updated` -> debounced
//! `SearchIndexManager` -> `upsert_session`.  The ignored benchmark below builds
//! a warm index for six sessions, appends one real ACP user update to each, then
//! measures the worker drain after the fixed debounce has elapsed.  The fixture
//! deliberately contains a multi-megabyte history of valid, non-indexable ACP
//! `available_commands_update` records: ordinary maintenance must not replay
//! that history merely to make one appended user message searchable.
//!
//! The non-ignored differential test compares the maintained FTS row against a
//! fresh full bootstrap for append, rewind, incomplete trailing-line, replaced,
//! truncated, missing, capped-content, title, and delete cases.  It is a
//! correctness guard for a cursor-backed implementation, not a timing test.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use agent_client_protocol as acp;
use tempfile::TempDir;
use xai_grok_shell::session::info::Info;
use xai_grok_shell::session::storage::search::{
    SessionSearchRequest, execute_search, notify_session_updated,
};
use xai_grok_shell::session::storage::search_fts::{SessionIndexState, SessionSearchIndex};
use xai_grok_shell::session::storage::{JsonlStorageAdapter, StorageAdapter};

const DISPATCH_SETTLE: Duration = Duration::from_millis(50);
const DEBOUNCE_SETTLE: Duration = Duration::from_millis(600);
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const WAIT_TIMEOUT: Duration = Duration::from_secs(15);

// The benchmark's measured shape is six sessions with about 4 MiB each of
// historical, non-indexable events and one appended user message per session.
const BENCH_SESSION_COUNT: usize = 6;
const BENCH_SMALL_NOISE_LINES: usize = 1;
const BENCH_NOISE_LINES: usize = 512;
const BENCH_NOISE_BYTES: usize = 8 * 1024;

#[derive(Clone)]
struct FixtureSession {
    info: Info,
    updates_path: PathBuf,
}

fn worktree_tempdir(label: &str) -> TempDir {
    let cwd = std::env::current_dir().expect("determine worktree root");
    let prefix = format!(".perfloop-session-search-{label}-");
    tempfile::Builder::new()
        .prefix(&prefix)
        .tempdir_in(cwd)
        .expect("create worktree-local fixture directory")
}

fn set_grok_home(root: &Path) {
    // Each sealed invocation runs one exact test in a new process, before the
    // search manager's lazy global can read this process-wide setting.
    unsafe {
        std::env::set_var("GROK_HOME", root);
    }
}

fn text_chunk(text: String) -> acp::ContentChunk {
    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(text)))
}

fn acp_line(session_id: &str, update: acp::SessionUpdate) -> String {
    let update = serde_json::to_value(update).expect("serialize ACP update");
    serde_json::to_string(&serde_json::json!({
        "timestamp": 0u64,
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    }))
    .expect("serialize update envelope")
}

fn user_line(session_id: &str, text: impl Into<String>) -> String {
    acp_line(
        session_id,
        acp::SessionUpdate::UserMessageChunk(text_chunk(text.into())),
    )
}

fn assistant_line(session_id: &str, text: impl Into<String>) -> String {
    acp_line(
        session_id,
        acp::SessionUpdate::AgentMessageChunk(text_chunk(text.into())),
    )
}

fn rewind_line(session_id: &str, target_prompt_index: usize) -> String {
    serde_json::to_string(&serde_json::json!({
        "timestamp": 0u64,
        "method": "_x.ai/session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "rewind_marker",
                "target_prompt_index": target_prompt_index,
                "created_at": "2026-01-01T00:00:00Z",
            },
        },
    }))
    .expect("serialize rewind envelope")
}

fn noise_line(session_id: &str, payload: &str, ordinal: usize) -> String {
    // Build the same typed ACP update production persists for a slash-command
    // catalog. Search must classify it, but it contributes no indexable text.
    let command =
        acp::AvailableCommand::new(format!("fixture-command-{ordinal}"), payload.to_owned()).input(
            Some(acp::AvailableCommandInput::Unstructured(
                acp::UnstructuredCommandInput::new("[fixture argument]".to_owned()),
            )),
        );
    acp_line(
        session_id,
        acp::SessionUpdate::AvailableCommandsUpdate(acp::AvailableCommandsUpdate::new(vec![
            command,
        ])),
    )
}

fn jsonl_bytes(lines: impl IntoIterator<Item = String>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for line in lines {
        bytes.extend_from_slice(line.as_bytes());
        bytes.push(b'\n');
    }
    bytes
}

fn repeated_text(prefix: &str, bytes: usize) -> String {
    let mut text = String::with_capacity(bytes);
    text.push_str(prefix);
    text.push('-');
    while text.len() < bytes {
        text.push_str("search projection state must preserve full-replay semantics ");
    }
    text.truncate(bytes);
    text
}

fn short_history(session_id: &str, label: &str) -> Vec<u8> {
    jsonl_bytes([
        user_line(session_id, format!("{label}-old-user")),
        assistant_line(session_id, format!("{label}-old-assistant")),
    ])
}

fn history_with_noise(
    session_id: &str,
    label: &str,
    lines: usize,
    payload_bytes: usize,
) -> Vec<u8> {
    let payload = repeated_text(&format!("{label}-noise"), payload_bytes);
    let mut records = Vec::with_capacity(lines + 2);
    records.push(user_line(session_id, format!("{label}-old-user")));
    records.push(assistant_line(session_id, format!("{label}-old-assistant")));
    for ordinal in 0..lines {
        records.push(noise_line(session_id, &payload, ordinal));
    }
    jsonl_bytes(records)
}

async fn create_session(
    adapter: &JsonlStorageAdapter,
    root: &Path,
    name: &str,
    updates: Option<&[u8]>,
) -> FixtureSession {
    let cwd = root.join("fixture-cwd");
    fs::create_dir_all(&cwd).expect("create fixture cwd");
    let info = Info {
        id: acp::SessionId::new(name.to_owned()),
        cwd: cwd.to_string_lossy().into_owned(),
    };
    adapter
        .init_session(&info, acp::ModelId::new("fixture-model"))
        .await
        .expect("initialize fixture session");
    let updates_path = adapter
        .updates_file_path(&info)
        .expect("JSONL adapter exposes updates path");
    if let Some(updates) = updates {
        fs::write(&updates_path, updates).expect("write fixture updates");
    }
    FixtureSession { info, updates_path }
}

fn read_state(root: &Path, session_id: &str) -> Option<SessionIndexState> {
    let index =
        SessionSearchIndex::open_or_create(&root.join("sessions/session_search.sqlite")).ok()?;
    index.get_session_index_state(session_id).ok().flatten()
}

fn title_is_indexed(root: &Path, session_id: &str, title_token: &str) -> bool {
    let Ok(index) =
        SessionSearchIndex::open_or_create(&root.join("sessions/session_search.sqlite"))
    else {
        return false;
    };
    index
        .query(title_token, None, 10, 0, false)
        .map(|result| {
            result
                .results
                .iter()
                .any(|row| row.session_id == session_id)
        })
        .unwrap_or(false)
}

async fn wait_until(label: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    loop {
        if condition() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {label} after {} seconds",
            WAIT_TIMEOUT.as_secs()
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn bootstrap(root: &Path, sessions: &[FixtureSession]) {
    let request = SessionSearchRequest {
        query: "fixture".to_owned(),
        cwd: None,
        limit: 10,
        offset: 0,
        include_content: false,
    };
    execute_search(root, &request)
        .await
        .expect("bootstrap session-search index");

    for session in sessions {
        let session_id = session.info.id.to_string();
        wait_until(&format!("initial index row for {session_id}"), || {
            read_state(root, &session_id).is_some()
        })
        .await;
    }
}

/// Let the manager receive all notifications, then hold the single-threaded
/// Tokio runtime past the debounce deadline.  The measured test starts its
/// timer immediately before yielding the worker, so the fixed 500 ms debounce
/// cannot dilute the worker-maintenance metric.
async fn queue_and_pass_debounce(sessions: &[FixtureSession]) {
    for session in sessions {
        notify_session_updated(&session.info.id.to_string(), &session.info.cwd);
    }
    tokio::time::sleep(DISPATCH_SETTLE).await;
    std::thread::sleep(DEBOUNCE_SETTLE);
    tokio::task::yield_now().await;
}

async fn append_user(adapter: &JsonlStorageAdapter, session: &FixtureSession, text: String) {
    adapter
        .append_update(
            &session.info,
            &acp::SessionUpdate::UserMessageChunk(text_chunk(text)),
        )
        .await
        .expect("append user update");
}

async fn append_assistant(adapter: &JsonlStorageAdapter, session: &FixtureSession, text: String) {
    adapter
        .append_update(
            &session.info,
            &acp::SessionUpdate::AgentMessageChunk(text_chunk(text)),
        )
        .await
        .expect("append assistant update");
}

async fn measure_worker_batch(
    adapter: &JsonlStorageAdapter,
    root: &Path,
    sessions: &[FixtureSession],
    token_prefix: &str,
) -> f64 {
    let mut tokens = Vec::with_capacity(sessions.len());
    for (index, session) in sessions.iter().enumerate() {
        let token = format!("{token_prefix}-{index}");
        append_user(adapter, session, token.clone()).await;
        tokens.push(token);
    }

    for session in sessions {
        notify_session_updated(&session.info.id.to_string(), &session.info.cwd);
    }
    tokio::time::sleep(DISPATCH_SETTLE).await;
    std::thread::sleep(DEBOUNCE_SETTLE);
    let started = Instant::now();
    tokio::task::yield_now().await;

    for (session, token) in sessions.iter().zip(&tokens) {
        let session_id = session.info.id.to_string();
        wait_until(&format!("benchmark index update for {session_id}"), || {
            read_state(root, &session_id).is_some_and(|state| state.content.contains(token))
        })
        .await;
    }
    started.elapsed().as_nanos() as f64 / sessions.len() as f64
}

fn append_raw(path: &Path, bytes: &[u8]) {
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open updates for append");
    file.write_all(bytes).expect("append raw update bytes");
    file.flush().expect("flush raw update bytes");
}

fn append_lines(path: &Path, lines: impl IntoIterator<Item = String>) {
    let mut bytes = Vec::new();
    for line in lines {
        bytes.extend_from_slice(line.as_bytes());
        bytes.push(b'\n');
    }
    append_raw(path, &bytes);
}

fn replace_atomically(path: &Path, bytes: &[u8]) {
    let replacement = path.with_extension("replacement");
    fs::write(&replacement, bytes).expect("write replacement updates file");
    fs::rename(&replacement, path).expect("atomically replace updates file");
}

fn assert_same_state(
    label: &str,
    actual_root: &Path,
    actual: &FixtureSession,
    reference_root: &Path,
    reference: &FixtureSession,
) {
    let actual_state = read_state(actual_root, &actual.info.id.to_string())
        .unwrap_or_else(|| panic!("missing actual state for {label}"));
    let reference_state = read_state(reference_root, &reference.info.id.to_string())
        .unwrap_or_else(|| panic!("missing full-replay reference state for {label}"));
    assert_eq!(
        actual_state.content, reference_state.content,
        "{label}: incremental content must equal a fresh full replay"
    );
    assert_eq!(
        actual_state.content_hash, reference_state.content_hash,
        "{label}: content hash must equal a fresh full replay"
    );
    assert_eq!(
        actual_state.last_indexed_offset, reference_state.last_indexed_offset,
        "{label}: durable cursor must end at the full-replay file boundary"
    );
}

/// A differential correctness guard for every recovery boundary a durable
/// cursor has to preserve.  The reference root is bootstrapped after all final
/// files are written; it is therefore the old, authoritative full-replay
/// behavior rather than a duplicated implementation in this test.
#[tokio::test(flavor = "current_thread")]
async fn session_search_incremental_matches_full_replay_across_recovery_paths() {
    let root = worktree_tempdir("correctness");
    set_grok_home(root.path());
    let adapter = JsonlStorageAdapter::with_root(root.path().to_path_buf());

    let ordinary = create_session(
        &adapter,
        root.path(),
        "ordinary",
        Some(&short_history("ordinary", "ordinary")),
    )
    .await;
    let rewind = create_session(
        &adapter,
        root.path(),
        "rewind",
        Some(&jsonl_bytes([
            user_line("rewind", "rewind-keep-user"),
            assistant_line("rewind", "rewind-keep-assistant"),
            user_line("rewind", "rewind-remove-user"),
            assistant_line("rewind", "rewind-remove-assistant"),
        ])),
    )
    .await;
    let partial = create_session(
        &adapter,
        root.path(),
        "partial",
        Some(&short_history("partial", "partial")),
    )
    .await;
    let replaced = create_session(
        &adapter,
        root.path(),
        "replaced",
        Some(&short_history("replaced", "replaced")),
    )
    .await;
    let truncated = create_session(
        &adapter,
        root.path(),
        "truncated",
        Some(&history_with_noise("truncated", "truncated", 8, 1024)),
    )
    .await;
    let missing = create_session(
        &adapter,
        root.path(),
        "missing",
        Some(&short_history("missing", "missing")),
    )
    .await;
    let capped = create_session(
        &adapter,
        root.path(),
        "capped",
        Some(&jsonl_bytes([
            user_line("capped", repeated_text("capped-old", 120_000)),
            assistant_line("capped", "capped-old-assistant"),
        ])),
    )
    .await;
    let title = create_session(
        &adapter,
        root.path(),
        "title",
        Some(&short_history("title", "title")),
    )
    .await;
    let deleted = create_session(
        &adapter,
        root.path(),
        "deleted",
        Some(&short_history("deleted", "deleted")),
    )
    .await;

    let all = vec![
        ordinary.clone(),
        rewind.clone(),
        partial.clone(),
        replaced.clone(),
        truncated.clone(),
        missing.clone(),
        capped.clone(),
        title.clone(),
        deleted.clone(),
    ];
    bootstrap(root.path(), &all).await;

    // First make a torn trailing line observable.  The temporary title change
    // lets the test wait for that worker pass without relying on cursor internals.
    let partial_complete = user_line("partial", "partial-complete-user");
    let split = partial_complete.len() / 2;
    adapter
        .update_session_title(&partial.info, "partial-intermediate-title".to_owned())
        .await
        .expect("update partial title");
    append_raw(&partial.updates_path, &partial_complete.as_bytes()[..split]);
    queue_and_pass_debounce(std::slice::from_ref(&partial)).await;
    wait_until("partial intermediate title indexed", || {
        title_is_indexed(
            root.path(),
            &partial.info.id.to_string(),
            "partial-intermediate-title",
        )
    })
    .await;

    append_user(&adapter, &ordinary, "ordinary-appended-user".to_owned()).await;
    append_assistant(
        &adapter,
        &ordinary,
        "ordinary-appended-assistant".to_owned(),
    )
    .await;

    append_lines(
        &rewind.updates_path,
        [
            rewind_line("rewind", 1),
            user_line("rewind", "rewind-replacement-user"),
            assistant_line("rewind", "rewind-replacement-assistant"),
        ],
    );

    // Complete exactly the formerly torn line.  A cursor that advanced across
    // its incomplete prefix would now skip this valid user message.
    append_raw(&partial.updates_path, &partial_complete.as_bytes()[split..]);
    append_raw(&partial.updates_path, b"\n");

    let replacement_final = history_with_noise("replaced", "replacement-new", 4, 1024);
    replace_atomically(&replaced.updates_path, &replacement_final);

    let truncation_final = short_history("truncated", "truncation-new");
    fs::write(&truncated.updates_path, &truncation_final).expect("truncate updates file");

    fs::remove_file(&missing.updates_path).expect("remove updates file");

    append_user(
        &adapter,
        &capped,
        repeated_text("capped-new-tail-token", 120_000),
    )
    .await;

    adapter
        .update_session_title(&title.info, "title-after-indexing".to_owned())
        .await
        .expect("update title-only session");

    adapter
        .delete_session(&deleted.info)
        .await
        .expect("delete session data");

    queue_and_pass_debounce(&all).await;
    wait_until("ordinary append", || {
        read_state(root.path(), "ordinary")
            .is_some_and(|state| state.content.contains("ordinary-appended-assistant"))
    })
    .await;
    wait_until("rewind replacement", || {
        read_state(root.path(), "rewind")
            .is_some_and(|state| state.content.contains("rewind-replacement-assistant"))
    })
    .await;
    wait_until("completed trailing line", || {
        read_state(root.path(), "partial")
            .is_some_and(|state| state.content.contains("partial-complete-user"))
    })
    .await;
    wait_until("replaced log", || {
        read_state(root.path(), "replaced")
            .is_some_and(|state| state.content.contains("replacement-new-old-user"))
    })
    .await;
    wait_until("truncated log", || {
        read_state(root.path(), "truncated")
            .is_some_and(|state| state.content.contains("truncation-new-old-user"))
    })
    .await;
    wait_until("missing log", || {
        read_state(root.path(), "missing").is_some_and(|state| state.content.is_empty())
    })
    .await;
    wait_until("capped projection tail", || {
        read_state(root.path(), "capped")
            .is_some_and(|state| state.content.contains("capped-new-tail-token"))
    })
    .await;
    wait_until("title-only update", || {
        title_is_indexed(root.path(), "title", "title-after-indexing")
    })
    .await;
    wait_until("deleted document removed", || {
        read_state(root.path(), "deleted").is_none()
    })
    .await;

    // Build a distinct, fresh root holding the final authoritative files.  Its
    // bootstrap is the reference result for every maintained row above.
    let reference_root = worktree_tempdir("reference");
    let reference_adapter = JsonlStorageAdapter::with_root(reference_root.path().to_path_buf());
    let ordinary_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "ordinary",
        Some(&fs::read(&ordinary.updates_path).expect("read ordinary final log")),
    )
    .await;
    let rewind_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "rewind",
        Some(&fs::read(&rewind.updates_path).expect("read rewind final log")),
    )
    .await;
    let partial_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "partial",
        Some(&fs::read(&partial.updates_path).expect("read partial final log")),
    )
    .await;
    reference_adapter
        .update_session_title(&partial_ref.info, "partial-intermediate-title".to_owned())
        .await
        .expect("set partial reference title");
    let replaced_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "replaced",
        Some(&fs::read(&replaced.updates_path).expect("read replacement final log")),
    )
    .await;
    let truncated_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "truncated",
        Some(&fs::read(&truncated.updates_path).expect("read truncation final log")),
    )
    .await;
    let missing_ref =
        create_session(&reference_adapter, reference_root.path(), "missing", None).await;
    let capped_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "capped",
        Some(&fs::read(&capped.updates_path).expect("read capped final log")),
    )
    .await;
    let title_ref = create_session(
        &reference_adapter,
        reference_root.path(),
        "title",
        Some(&fs::read(&title.updates_path).expect("read title final log")),
    )
    .await;
    reference_adapter
        .update_session_title(&title_ref.info, "title-after-indexing".to_owned())
        .await
        .expect("set title reference title");

    let references = vec![
        ordinary_ref.clone(),
        rewind_ref.clone(),
        partial_ref.clone(),
        replaced_ref.clone(),
        truncated_ref.clone(),
        missing_ref.clone(),
        capped_ref.clone(),
        title_ref.clone(),
    ];
    bootstrap(reference_root.path(), &references).await;

    assert_same_state(
        "ordinary append",
        root.path(),
        &ordinary,
        reference_root.path(),
        &ordinary_ref,
    );
    assert_same_state(
        "rewind fallback",
        root.path(),
        &rewind,
        reference_root.path(),
        &rewind_ref,
    );
    assert_same_state(
        "completed trailing line",
        root.path(),
        &partial,
        reference_root.path(),
        &partial_ref,
    );
    assert_same_state(
        "replaced log fallback",
        root.path(),
        &replaced,
        reference_root.path(),
        &replaced_ref,
    );
    assert_same_state(
        "truncated log fallback",
        root.path(),
        &truncated,
        reference_root.path(),
        &truncated_ref,
    );
    assert_same_state(
        "missing log fallback",
        root.path(),
        &missing,
        reference_root.path(),
        &missing_ref,
    );
    assert_same_state(
        "capped projection",
        root.path(),
        &capped,
        reference_root.path(),
        &capped_ref,
    );
    assert_same_state(
        "title-only update",
        root.path(),
        &title,
        reference_root.path(),
        &title_ref,
    );
    assert!(
        read_state(root.path(), "deleted").is_none(),
        "deleted session must not remain in the maintained index"
    );
}

/// One per-invocation sample for the proof controller.  The fixture is created
/// and fully bootstrapped outside the timers.  It measures the same six-session
/// append batch at a tiny and a roughly 4 MiB/session prior-history point.  The
/// primary large-history metric demonstrates the user-visible maintenance cost;
/// the small-history and byte metrics make the algorithmic input-size sweep
/// explicit without turning that supporting comparison into a second goal.
#[tokio::test(flavor = "current_thread")]
#[ignore = "performance sample; run explicitly with --ignored --nocapture"]
async fn session_search_worker_large_history_batch() {
    let root = worktree_tempdir("benchmark");
    set_grok_home(root.path());
    let adapter = JsonlStorageAdapter::with_root(root.path().to_path_buf());

    let mut small_sessions = Vec::with_capacity(BENCH_SESSION_COUNT);
    let mut large_sessions = Vec::with_capacity(BENCH_SESSION_COUNT);
    for index in 0..BENCH_SESSION_COUNT {
        let small_name = format!("benchmark-small-{index}");
        let small_initial = history_with_noise(
            &small_name,
            &small_name,
            BENCH_SMALL_NOISE_LINES,
            BENCH_NOISE_BYTES,
        );
        small_sessions
            .push(create_session(&adapter, root.path(), &small_name, Some(&small_initial)).await);

        let large_name = format!("benchmark-large-{index}");
        let large_initial = history_with_noise(
            &large_name,
            &large_name,
            BENCH_NOISE_LINES,
            BENCH_NOISE_BYTES,
        );
        large_sessions
            .push(create_session(&adapter, root.path(), &large_name, Some(&large_initial)).await);
    }
    let mut all_sessions = small_sessions.clone();
    all_sessions.extend(large_sessions.clone());
    bootstrap(root.path(), &all_sessions).await;

    let small_history_bytes_per_upsert = small_sessions
        .iter()
        .map(|session| {
            fs::metadata(&session.updates_path)
                .expect("stat small fixture history")
                .len()
        })
        .sum::<u64>() as f64
        / BENCH_SESSION_COUNT as f64;
    let history_bytes_per_upsert = large_sessions
        .iter()
        .map(|session| {
            fs::metadata(&session.updates_path)
                .expect("stat large fixture history")
                .len()
        })
        .sum::<u64>() as f64
        / BENCH_SESSION_COUNT as f64;
    assert!(
        history_bytes_per_upsert >= small_history_bytes_per_upsert * 100.0,
        "fixture must retain a meaningful prior-history sweep"
    );

    // Run the large point first so the primary sample is not helped by any
    // immediately preceding small-batch worker activity.  Both batches use the
    // same real debounce, worker, storage, and FTS state transition.
    let ns_per_upsert = measure_worker_batch(
        &adapter,
        root.path(),
        &large_sessions,
        "benchmark-large-appended-token",
    )
    .await;
    let small_ns_per_upsert = measure_worker_batch(
        &adapter,
        root.path(),
        &small_sessions,
        "benchmark-small-appended-token",
    )
    .await;

    // These are the only JSON objects the sealed adapter forwards.  The result
    // is observed through `get_session_index_state` in `measure_worker_batch`,
    // so neither the worker update nor the appended searchable token can be
    // optimized away.
    println!(
        "{}",
        serde_json::json!({
            "metric": "session_search_worker_ns_per_upsert",
            "value": ns_per_upsert,
        })
    );
    println!(
        "{}",
        serde_json::json!({
            "metric": "session_search_worker_small_history_ns_per_upsert",
            "value": small_ns_per_upsert,
        })
    );
    println!(
        "{}",
        serde_json::json!({
            "metric": "session_search_history_bytes_per_upsert",
            "value": history_bytes_per_upsert,
        })
    );
    println!(
        "{}",
        serde_json::json!({
            "metric": "session_search_small_history_bytes_per_upsert",
            "value": small_history_bytes_per_upsert,
        })
    );
}
