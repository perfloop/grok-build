//! Regression and allocation tests for the session-to-sampler request handoff.
//!
//! The allocation probe deliberately blocks the mock response immediately before
//! its terminal SSE event. At that point the session request is still in flight,
//! so DHAT's current-byte snapshot sees the payload copies retained by the
//! handoff rather than transient serialization buffers.

use super::support::*;
use super::*;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;
use xai_grok_sampler::{ApiBackend, RetryPolicy, SamplerActor, SamplingEvent};
use xai_grok_test_support::{
    InferenceEndpoint, InferenceRequestMatcher, MockInferenceServer, ScriptedResponse, SseEvent,
};

const REQUEST_MARKER: &str = "request-move-large-tool-schema";
const USER_MARKER: &str = "request-move-user-input";
const REQUEST_SCHEMA_BYTES: usize = 1_048_576;
const LIVE_BYTES_METRIC: &str = "turn_live_request_bytes";

fn large_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "request_move_tool".to_string(),
        description: Some(format!("fixture {REQUEST_MARKER}")),
        parameters: json!({
            "type": "object",
            "properties": {
                "payload": {
                    "type": "string",
                    "description": "x".repeat(REQUEST_SCHEMA_BYTES),
                    "fixture_marker": REQUEST_MARKER,
                },
            },
        }),
    }
}

fn successful_chat_stream() -> ScriptedResponse {
    ScriptedResponse::sse(vec![
        SseEvent::data(
            json!({
                "id": "chatcmpl-request-move",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "request-move-model",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "request accepted"},
                    "finish_reason": null,
                }],
            })
            .to_string(),
        ),
        SseEvent::data(
            json!({
                "id": "chatcmpl-request-move",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "request-move-model",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            })
            .to_string(),
        ),
        SseEvent::data("[DONE]"),
    ])
}

async fn actor_with_real_sampler(server: &MockInferenceServer) -> Arc<SessionActor> {
    let (gateway_tx, _gateway_rx) = mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
    let (persistence_tx, _persistence_rx) = mpsc::unbounded_channel::<PersistenceMsg>();
    let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;

    let mut sampling_config = actor
        .chat_state_handle
        .get_sampling_config()
        .await
        .expect("test chat state is available");
    sampling_config.base_url = server.url();
    sampling_config.model = "request-move-model".to_string();
    sampling_config.api_backend = ApiBackend::ChatCompletions;
    actor
        .chat_state_handle
        .update_sampling_config(sampling_config);
    actor
        .chat_state_handle
        .get_sampling_config()
        .await
        .expect("sampling configuration update is processed");

    actor.forked_tool_override = Some(vec![large_tool_spec()]);
    let sampler_config = actor.reconstruct_full_config().await;
    let (sampler_event_tx, mut sampler_event_rx) = mpsc::unbounded_channel::<SamplingEvent>();
    actor.sampler_handle =
        SamplerActor::spawn(sampler_config, RetryPolicy::default(), sampler_event_tx);
    actor
        .chat_state_handle
        .push_user_message_and_ack(ConversationItem::user(USER_MARKER))
        .await;

    let actor = Arc::new(actor);
    let event_actor = Arc::clone(&actor);
    let _event_drainer = tokio::task::spawn_local(async move {
        while let Some(event) = sampler_event_rx.recv().await {
            event_actor.handle_sampling_event(event).await;
        }
    });
    actor
}

async fn run_turn(actor: Arc<SessionActor>) -> Result<TurnOutcome, acp::Error> {
    actor
        .process_conversation_turn_with_recovery("request-move-turn", None, None, None)
        .await
}

#[tokio::test(flavor = "current_thread")]
async fn session_to_sampler_handoff_preserves_populated_request() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let server = MockInferenceServer::start()
                .await
                .expect("mock inference server starts");
            let mut expected = server.expect_response(
                "populated chat-completions handoff",
                InferenceRequestMatcher::foreground(InferenceEndpoint::ChatCompletions),
                successful_chat_stream(),
            );
            let actor = actor_with_real_sampler(&server).await;

            run_turn(actor)
                .await
                .expect("sampler completes the populated turn");
            expected.wait_satisfied().await;
            expected.assert_satisfied();

            let body = server
                .request_bodies()
                .into_iter()
                .find(|body| body["model"] == "request-move-model")
                .expect("session submits the request to the chat-completions endpoint");
            let wire = serde_json::to_string(&body).expect("request body serializes");
            assert!(
                wire.contains(REQUEST_MARKER),
                "the populated tool schema reaches the sampler wire request"
            );
            assert!(
                wire.contains(USER_MARKER),
                "the populated conversation reaches the sampler wire request"
            );
        })
        .await;
}

#[cfg(feature = "dhat-heap")]
#[tokio::test(flavor = "current_thread")]
async fn sampler_handoff_live_request_bytes() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let server = MockInferenceServer::start()
                .await
                .expect("mock inference server starts");
            let mut expected = server.expect_response_blocked(
                "blocked populated chat-completions handoff",
                InferenceRequestMatcher::foreground(InferenceEndpoint::ChatCompletions),
                successful_chat_stream(),
            );
            let actor = actor_with_real_sampler(&server).await;
            let _profiler = dhat::Profiler::builder().testing().build();

            let turn = tokio::task::spawn_local(run_turn(actor));
            expected.wait_blocked().await;

            let stats = dhat::HeapStats::get();
            let sample = json!({
                "metric": LIVE_BYTES_METRIC,
                "value": stats.curr_bytes,
                "unit": "bytes",
                "payload_bytes": REQUEST_SCHEMA_BYTES,
            });
            eprintln!("DHAT_REQUEST_MOVE_SUMMARY {sample}");
            if let Ok(result_path) = std::env::var("PERFLOOP_RESULT_FILE") {
                std::fs::write(result_path, format!("{sample}\n"))
                    .expect("write Perfloop allocation sample");
            }

            expected.release();
            turn.await
                .expect("turn task does not panic")
                .expect("sampler completes after releasing the terminal barrier");
            expected.wait_satisfied().await;
            expected.assert_satisfied();
        })
        .await;
}
