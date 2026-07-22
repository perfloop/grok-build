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
use serde_json::{Value, json};
use xai_grok_test_support::{
    GrokStdioClient, InferenceEndpoint, InferenceExpectation, InferenceRequestMatcher,
    MockInferenceServer, MockModelEntry, ScriptedResponse, SseEvent, git_workdir,
};

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
        // Keep fallback replies short so the workload cost is request history,
        // not SSE output generation. Foreground turn replies are registered
        // below with unique terminal response IDs.
        server.set_response("continuation benchmark acknowledgement");
        server
    }
}

fn response_id(turn: usize) -> String {
    format!("resp_continuation_{turn:02}")
}

/// Build a complete, typed Responses SSE stream with a caller-provided terminal
/// response ID. The next cursor request must name this exact ID, proving the
/// terminal stream value was propagated rather than fabricated locally.
fn responses_script(response_id: &str, model: &str) -> ScriptedResponse {
    const TEXT: &str = "continuation benchmark acknowledgement";
    ScriptedResponse::sse(vec![
        SseEvent::data(
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": 1234567890,
                    "model": model,
                    "status": "in_progress",
                    "output": [],
                },
            })
            .to_string(),
        ),
        SseEvent::data(
            json!({
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": format!("item_{response_id}"),
                "output_index": 0,
                "content_index": 0,
                "delta": TEXT,
            })
            .to_string(),
        ),
        SseEvent::data(
            json!({
                "type": "response.completed",
                "sequence_number": 2,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": 1234567890,
                    "model": model,
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": format!("msg_{response_id}"),
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": TEXT,
                            "annotations": [],
                        }],
                    }],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "input_tokens_details": { "cached_tokens": 0 },
                        "output_tokens_details": { "reasoning_tokens": 0 },
                    },
                },
            })
            .to_string(),
        ),
        SseEvent::data("[DONE]"),
    ])
}

fn expect_responses(
    server: &MockInferenceServer,
    model: &str,
    response_ids: impl IntoIterator<Item = String>,
) -> Vec<InferenceExpectation> {
    response_ids
        .into_iter()
        .enumerate()
        .map(|(turn, response_id)| {
            server.expect_response(
                format!("Responses continuation turn {turn}"),
                InferenceRequestMatcher::foreground(InferenceEndpoint::Responses),
                responses_script(&response_id, model),
            )
        })
        .collect()
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
/// future continuation implementation. A warm request must name the exact
/// unique ID emitted by the immediately preceding terminal SSE frame; a bogus
/// locally fabricated cursor cannot satisfy this chain.
fn assert_history_or_cursor_chain(turns: &[Value]) {
    assert_eq!(
        turns
            .first()
            .and_then(|body| body.get("previous_response_id")),
        None,
        "the first Responses request must seed from local history"
    );

    let uses_cursor = turns
        .iter()
        .skip(1)
        .any(|body| body.get("previous_response_id").is_some());
    if !uses_cursor {
        assert!(
            turns
                .iter()
                .all(|body| body.get("previous_response_id").is_none()),
            "fallback requests must not mix local history with an unverified cursor"
        );
        let final_items = input_items(turns.last().expect("final request"));
        for turn in 0..SEED_TURNS {
            assert!(
                contains_text(final_items, &seed_marker(turn)),
                "without a continuation cursor, the final request must retain every local seed"
            );
        }
        return;
    }

    assert_eq!(
        turns[0].get("store"),
        Some(&Value::Bool(true)),
        "the seed response must be stored before a later request can continue it"
    );
    for (turn, body) in turns.iter().enumerate().skip(1) {
        let expected_cursor = response_id(turn - 1);
        assert_eq!(
            body.get("previous_response_id").and_then(Value::as_str),
            Some(expected_cursor.as_str()),
            "warm request {turn} must continue the immediately preceding terminal response"
        );
        assert_eq!(
            body.get("store"),
            Some(&Value::Bool(true)),
            "every continuation response must remain stored for the next cursor"
        );

        let items = input_items(body);
        if turn < SEED_TURNS {
            assert!(
                contains_text(items, &seed_marker(turn)),
                "warm seed request {turn} must retain its newly appended ACP prompt"
            );
            assert!(
                !contains_text(items, &seed_marker(turn - 1)),
                "warm seed request {turn} must not resend its predecessor's history"
            );
        } else {
            assert!(
                !contains_text(items, SEED_MARKER),
                "the final cursor request must not resend any seeded local history"
            );
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
        let expectations = expect_responses(&server, MODEL_A, (0..=SEED_TURNS).map(response_id));
        let workdir = git_workdir();
        let client = GrokStdioClient::spawn(&server, workdir.path()).await;
        client.initialize().await;
        let session_id = client
            .create_session_with_model(workdir.path(), MODEL_A)
            .await;

        for turn in 0..SEED_TURNS {
            let prompt = seed_prompt(turn);
            let response = client
                .prompt(&session_id, &prompt)
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
            .prompt(&session_id, FINAL_PROMPT)
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
        assert_history_or_cursor_chain(&turns);
        for expectation in &expectations {
            expectation.assert_satisfied();
        }
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
        let before_switch = server.expect_response(
            "Responses response before model switch",
            InferenceRequestMatcher::foreground(InferenceEndpoint::Responses),
            responses_script("resp_before_model_switch", MODEL_A),
        );
        let after_switch_response = server.expect_response(
            "Responses response after model switch",
            InferenceRequestMatcher::foreground(InferenceEndpoint::Responses),
            responses_script("resp_after_model_switch", MODEL_B),
        );
        let workdir = git_workdir();
        let client = GrokStdioClient::spawn(&server, workdir.path()).await;
        client.initialize().await;
        let session_id = client
            .create_session_with_model(workdir.path(), MODEL_A)
            .await;

        let seed = seed_prompt(0);
        client
            .prompt(&session_id, &seed)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "seed prompt failed: {error:?}\nstderr:\n{}",
                    client.stderr()
                )
            });
        let switch = client.set_model(&session_id, MODEL_B).await;
        assert!(
            switch.is_ok(),
            "same-harness Responses model switch should succeed: {switch:?}\nstderr:\n{}",
            client.stderr()
        );
        let response = client
            .prompt(&session_id, FINAL_PROMPT)
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
        before_switch.assert_satisfied();
        after_switch_response.assert_satisfied();
    })
    .await;
}
