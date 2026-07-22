//! End-to-end proof surface for Responses continuation requests.
//!
//! The workload is deliberately a long local-history session: sixteen 16 KiB
//! user prompts followed by a short ACP prompt. Each seed stays below the
//! shell's individual-prompt offload threshold, so the final request must
//! either retain the accumulated local history in a full snapshot or carry an
//! explicit server cursor. This lets the test remain valid before and after a
//! continuation implementation.
//!
//! The second test makes model changes a hard reset boundary: stale remote
//! context must never cross a model switch.
//!
//! These tests require the composed pager binary and are ignored by default:
//! ```sh
//! cargo test -p xai-grok-shell --test responses_continuation_bench -- --ignored
//! ```

use std::future::Future;

use agent_client_protocol as acp;
use serde_json::Value;
use xai_grok_test_support::{GrokStdioClient, MockInferenceServer, MockModelEntry, git_workdir};

const MODEL_A: &str = "responses-continuation-a";
const MODEL_B: &str = "responses-continuation-b";
const SEED_TURNS: usize = 16;
const SEED_PROMPT_BYTES: usize = 16 * 1024;
const SEED_MARKER: &str = "responses-continuation-seed-marker";
const FINAL_PROMPT: &str = "Summarize the prior request in one sentence.";

/// ACP's client-side connection owns `!Send` futures, so drive it on a local
/// task set exactly as the existing end-to-end suites do.
async fn with_local_set<F, Fut>(f: F)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    tokio::task::LocalSet::new().run_until(f()).await;
}

fn seed_marker(turn: usize) -> String {
    format!("{SEED_MARKER}-{turn:02}")
}

fn seed_prompt(turn: usize) -> String {
    let mut prompt = format!("{}\n", seed_marker(turn));
    prompt.reserve(SEED_PROMPT_BYTES - prompt.len());
    while prompt.len() < SEED_PROMPT_BYTES {
        prompt.push_str("retain this seeded request context across later prompts; ");
    }
    prompt.truncate(SEED_PROMPT_BYTES);
    prompt
}

fn responses_server(models: Vec<MockModelEntry>) -> impl Future<Output = MockInferenceServer> {
    async move {
        let server = MockInferenceServer::start_with_models(models)
            .await
            .expect("start Responses mock server");
        // Keep replies short so the workload cost is request history, not SSE
        // output generation. The terminal response still carries a response id.
        server.set_response("continuation benchmark acknowledgement");
        server
    }
}

fn turn_bodies(server: &MockInferenceServer) -> Vec<Value> {
    server
        .requests()
        .into_iter()
        .filter(|entry| entry.method == "POST" && entry.path == "/v1/responses")
        // Auxiliary work such as title generation can also use Responses. The
        // turn index is assigned only by process_conversation_turn.
        .filter(|entry| entry.header("x-grok-turn-idx").is_some())
        .filter_map(|entry| entry.body)
        .collect()
}

fn input_items(body: &Value) -> &[Value] {
    body.get("input")
        .and_then(Value::as_array)
        .expect("Responses turn request must carry an input array")
}

fn item_text(item: &Value) -> Option<&str> {
    item.get("content").and_then(|content| {
        content.as_str().or_else(|| {
            content.as_array().and_then(|parts| {
                parts.iter().find_map(|part| {
                    (part.get("type").and_then(Value::as_str) == Some("input_text"))
                        .then(|| part.get("text").and_then(Value::as_str))
                        .flatten()
                })
            })
        })
    })
}

fn contains_text(items: &[Value], expected: &str) -> bool {
    items
        .iter()
        .filter_map(item_text)
        // The shell can wrap ACP prompts in an on-disk prompt-file envelope.
        // The original user text must therefore be present, but need not be
        // the whole input-text block.
        .any(|text| text.contains(expected))
}

fn latest_user_text(items: &[Value]) -> Option<&str> {
    items
        .iter()
        .rev()
        .find(|item| item.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(item_text)
}

/// Assert the protocol contract shared by the full-snapshot baseline and a
/// future continuation implementation. A delta without a predecessor id would
/// silently discard user context; a cursor without storage would not be a
/// durable remote checkpoint.
fn assert_history_or_cursor(body: &Value) {
    let items = input_items(body);
    match body.get("previous_response_id").and_then(Value::as_str) {
        Some(cursor) => {
            assert!(!cursor.is_empty(), "continuation cursor must be nonempty");
            assert_eq!(
                body.get("store"),
                Some(&Value::Bool(true)),
                "a continuation cursor needs a stored remote checkpoint"
            );
            assert!(
                !contains_text(items, SEED_MARKER),
                "a cursor request should send only the newly appended delta, not the full seed"
            );
        }
        None => {
            for turn in 0..SEED_TURNS {
                assert!(
                    contains_text(items, &seed_marker(turn)),
                    "without a continuation cursor, the request must retain every local seed"
                );
            }
        }
    }
}

/// Measure the final Responses request after sixteen 16 KiB seed prompts.
/// The emitted byte count is the JSON body observed at the HTTP boundary;
/// input-item count is a supporting structural signal.
#[tokio::test]
#[ignore = "requires the composed xai-grok-pager binary"]
async fn responses_continuation_long_history_final_turn() {
    with_local_set(|| async {
        let server = responses_server(vec![
            MockModelEntry::new(MODEL_A).with_api_backend("responses"),
        ])
        .await;
        let workdir = git_workdir();
        let client = GrokStdioClient::spawn(&server, workdir.path()).await;
        client.initialize_with_timeout().await;
        let session_id = client
            .create_session_with_model_timeout(workdir.path(), MODEL_A)
            .await;

        for turn in 0..SEED_TURNS {
            let prompt = seed_prompt(turn);
            let response = client
                .prompt_with_timeout(&session_id, &prompt)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "seed turn {turn} failed: {error:?}\nstderr:\n{}",
                        client.stderr()
                    )
                });
            assert_eq!(response.stop_reason, acp::StopReason::EndTurn);
        }
        let response = client
            .prompt_with_timeout(&session_id, FINAL_PROMPT)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "final Responses prompt failed: {error:?}\nstderr:\n{}",
                    client.stderr()
                )
            });
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);

        let turns = turn_bodies(&server);
        assert_eq!(
            turns.len(),
            SEED_TURNS + 1,
            "expected one Responses turn request per ACP prompt\nrequest log:\n{}",
            server.request_log_summary()
        );
        assert!(
            contains_text(input_items(&turns[0]), &seed_marker(0)),
            "the initial seed must reach the first Responses request"
        );

        let final_body = turns.last().expect("seed and final turn requests");
        let final_items = input_items(final_body);
        assert!(
            latest_user_text(final_items).is_some_and(|text| text.contains(FINAL_PROMPT)),
            "the final request must contain the final ACP prompt"
        );
        assert_history_or_cursor(final_body);
        assert!(
            client
                .captured_text()
                .contains("continuation benchmark acknowledgement"),
            "the streamed Responses reply must still reach ACP"
        );

        let request_bytes = serde_json::to_vec(final_body)
            .expect("serialize captured Responses request")
            .len();
        println!(
            "PERFLOOP_JSON:{}",
            serde_json::json!({
                "metric": "responses_continuation_request_bytes",
                "value": request_bytes,
            })
        );
        println!(
            "PERFLOOP_JSON:{}",
            serde_json::json!({
                "metric": "responses_continuation_input_items",
                "value": final_items.len(),
            })
        );
    })
    .await;
}

/// A model switch changes the remote context contract. It must force a full
/// local-history reseed rather than attach a cursor minted under the old model.
#[tokio::test]
#[ignore = "requires the composed xai-grok-pager binary"]
async fn responses_continuation_model_switch_reseeds_history() {
    with_local_set(|| async {
        let server = responses_server(vec![
            MockModelEntry::new(MODEL_A).with_api_backend("responses"),
            MockModelEntry::new(MODEL_B).with_api_backend("responses"),
        ])
        .await;
        let workdir = git_workdir();
        let client = GrokStdioClient::spawn(&server, workdir.path()).await;
        client.initialize_with_timeout().await;
        let session_id = client
            .create_session_with_model_timeout(workdir.path(), MODEL_A)
            .await;

        let seed = seed_prompt(0);
        client
            .prompt_with_timeout(&session_id, &seed)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "seed prompt failed: {error:?}\nstderr:\n{}",
                    client.stderr()
                )
            });
        let switch = client.set_model_with_timeout(&session_id, MODEL_B).await;
        assert!(
            switch.is_ok(),
            "same-harness Responses model switch should succeed: {switch:?}\nstderr:\n{}",
            client.stderr()
        );
        let response = client
            .prompt_with_timeout(&session_id, FINAL_PROMPT)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "post-switch prompt failed: {error:?}\nstderr:\n{}",
                    client.stderr()
                )
            });
        assert_eq!(response.stop_reason, acp::StopReason::EndTurn);

        let turns = turn_bodies(&server);
        assert_eq!(
            turns.len(),
            2,
            "expected one turn before and after model switch\nrequest log:\n{}",
            server.request_log_summary()
        );
        let after_switch = turns.last().expect("post-switch request");
        assert!(
            after_switch.get("previous_response_id").is_none(),
            "model switch must invalidate any remote continuation cursor"
        );
        assert!(
            contains_text(input_items(after_switch), &seed_marker(0)),
            "model switch must reseed from authoritative local history"
        );
    })
    .await;
}
