//! Verdict-visible guard for the receiver and coalescing loop in `run_worker`.
//!
//! It shares the large-history fixture helpers with the primary maintenance
//! benchmark, but drives jobs through the worker's real channel. The
//! cfg(test)-only command explicitly advances the pending debounce and sends its
//! acknowledgement only after `flush_ready` has completed every upsert.

use std::path::Path;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use super::super::jsonl::JsonlStorageAdapter;
use super::search_maintenance_perf::{
    BENCH_NOISE_BYTES, BENCH_NOISE_LINES, BENCH_SESSION_COUNT, FixtureSession, append_user,
    assert_after_flush, create_session, history_with_noise, index_sessions, read_state,
    worktree_tempdir,
};
use super::{SearchIndexJob, SessionSearchKey, run_worker};

/// Exercise `run_worker` itself without a wall-clock wait. Jobs enter its real
/// receiver/coalescing loop in FIFO order, then the cfg(test) command advances
/// those pending deadlines and acknowledges only after `flush_ready` returns.
async fn flush_sessions_through_worker(root: &Path, sessions: &[FixtureSession]) {
    let (tx, rx) = mpsc::unbounded_channel();
    for session in sessions {
        let job = SearchIndexJob::Upsert(SessionSearchKey {
            session_id: session.info.id.to_string(),
            cwd: session.info.cwd.clone(),
        });
        tx.send(job)
            .unwrap_or_else(|_| panic!("test worker receiver unexpectedly closed"));
    }
    let (acknowledge, acknowledged) = oneshot::channel();
    tx.send(SearchIndexJob::FlushAndAcknowledge(acknowledge))
        .unwrap_or_else(|_| panic!("test worker receiver unexpectedly closed"));

    let worker_root = root.to_path_buf();
    let worker_storage = JsonlStorageAdapter::with_root(worker_root.clone());
    let worker = tokio::spawn(async move {
        run_worker(&worker_root, &worker_storage, rx).await;
    });

    acknowledged
        .await
        .expect("test worker must acknowledge the completed flush");
    drop(tx);
    worker
        .await
        .expect("test worker must stop after sender drop");
}

/// A verdict-visible guard for the receiver/coalescing portion of the anchored
/// worker. It uses the same large history as the primary selector but drives
/// `run_worker` through its real channel and receives the test-only completion
/// acknowledgement rather than waiting for wall-clock debounce.
#[tokio::test(flavor = "current_thread")]
#[ignore = "performance guard; run explicitly with --ignored --nocapture"]
async fn session_search_run_worker_large_history_batch() {
    let root = worktree_tempdir("worker-guard");
    let adapter = JsonlStorageAdapter::with_root(root.path().to_path_buf());
    let mut sessions = Vec::with_capacity(BENCH_SESSION_COUNT);
    for index in 0..BENCH_SESSION_COUNT {
        let name = format!("worker-guard-{index}");
        let initial = history_with_noise(&name, &name, BENCH_NOISE_LINES, BENCH_NOISE_BYTES);
        sessions.push(create_session(&adapter, root.path(), &name, Some(&initial)).await);
    }
    index_sessions(root.path(), &adapter, &sessions).await;

    let mut tokens = Vec::with_capacity(sessions.len());
    for (index, session) in sessions.iter().enumerate() {
        let token = format!("worker-guard-appended-token-{index}");
        append_user(&adapter, session, token.clone()).await;
        tokens.push(token);
    }

    let started = Instant::now();
    flush_sessions_through_worker(root.path(), &sessions).await;
    let ns_per_upsert = started.elapsed().as_nanos() as f64 / sessions.len() as f64;

    for (session, token) in sessions.iter().zip(&tokens) {
        let session_id = session.info.id.to_string();
        assert_after_flush(
            &format!("worker guard index update for {session_id}"),
            || {
                read_state(root.path(), &session_id)
                    .is_some_and(|state| state.content.contains(token))
            },
        )
        .await;
    }

    println!(
        "{}",
        serde_json::json!({
            "metric": "session_search_run_worker_ns_per_upsert",
            "value": ns_per_upsert,
        })
    );
}
