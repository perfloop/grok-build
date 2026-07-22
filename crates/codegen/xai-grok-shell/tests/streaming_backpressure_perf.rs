//! End-to-end slow-ACP-consumer workload for the streaming backpressure path.
//!
//! A loopback model emits a long Chat Completions response while the ACP
//! consumer accepts one outbound message only after a fixed delay. The workload
//! uses the real `MvpAgent -> SessionActor -> SamplerActor -> drive_l2` path,
//! records per-text-message enqueue-to-delivery age from production metadata,
//! and proves every streamed byte and terminal prompt response survive in order.
//!
//! Run:
//!   cargo test --release -p xai-grok-shell --test streaming_backpressure_perf -- --exact stalled_acp_streaming_reports_backlog_and_preserves_output --nocapture

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use agent_client_protocol::{self as acp, Agent as _};
use serde_json::json;
use tempfile::TempDir;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use xai_acp_lib::{AcpAgentGatewaySender as GatewaySender, AcpClientMessage, LineBufferedRead};
use xai_grok_shell::agent::config::Config as AgentConfig;
use xai_grok_shell::agent::mvp_agent::MvpAgent;
use xai_grok_test_support::MockInferenceServer;

const DEFAULT_STREAM_CHUNKS: usize = 4_096;
const DEFAULT_CHUNK_PAYLOAD_BYTES: usize = 4 * 1024;
const DEFAULT_CONSUMER_DELAY: Duration = Duration::from_millis(1);
const DUPLEX_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// The normal client-side RPC dispatcher is intentionally not installed for
/// this workload. The agent still receives initialize/new-session/prompt
/// requests over the duplex connection, while this test consumes its real
/// outbound gateway at a deliberately slower fixed rate.
struct NoopClient;

#[async_trait::async_trait(?Send)]
impl acp::Client for NoopClient {
    async fn request_permission(
        &self,
        args: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        let outcome = args
            .options
            .iter()
            .find(|option| option.kind == acp::PermissionOptionKind::AllowOnce)
            .or(args.options.first())
            .map(|option| {
                acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(
                    option.option_id.clone(),
                ))
            })
            .unwrap_or(acp::RequestPermissionOutcome::Cancelled);
        Ok(acp::RequestPermissionResponse::new(outcome))
    }

    async fn session_notification(&self, _args: acp::SessionNotification) -> acp::Result<()> {
        Ok(())
    }
}

struct Workload {
    chunks: usize,
    chunk_payload_bytes: usize,
    consumer_delay: Duration,
}

impl Workload {
    fn from_env() -> Self {
        fn positive_usize(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|value: &usize| *value > 0)
                .unwrap_or(default)
        }

        Self {
            chunks: positive_usize("GROK_PERF_STREAM_CHUNKS", DEFAULT_STREAM_CHUNKS),
            chunk_payload_bytes: positive_usize(
                "GROK_PERF_CHUNK_PAYLOAD_BYTES",
                DEFAULT_CHUNK_PAYLOAD_BYTES,
            ),
            consumer_delay: Duration::from_millis(positive_usize(
                "GROK_PERF_CONSUMER_DELAY_MS",
                DEFAULT_CONSUMER_DELAY.as_millis() as usize,
            ) as u64),
        }
    }
}

fn streamed_text(workload: &Workload) -> String {
    let payload = "x".repeat(workload.chunk_payload_bytes);
    let mut text = String::with_capacity(workload.chunks * (workload.chunk_payload_bytes + 16));
    for index in 0..workload.chunks {
        if index != 0 {
            text.push(' ');
        }
        write!(text, "chunk-{index:04}-").expect("write into String");
        text.push_str(&payload);
    }
    text
}

/// Consume one outbound ACP message. `agentTimestampMs` is produced by
/// `SessionActor::send_update` immediately before it queues the notification,
/// so comparing it to the receive time measures real in-process delivery age.
fn take_agent_text(
    message: AcpClientMessage,
    output: &mut String,
    chunks: &mut usize,
    delivery_ages_ms: &mut Vec<u64>,
) {
    let AcpClientMessage::SessionNotification(args) = message else {
        return;
    };
    let agent_timestamp_ms = args
        .request
        .meta
        .as_ref()
        .and_then(|meta| meta.get("agentTimestampMs"))
        .and_then(serde_json::Value::as_i64);
    let acp::SessionUpdate::AgentMessageChunk(chunk) = args.request.update else {
        return;
    };
    let acp::ContentBlock::Text(text) = chunk.content else {
        return;
    };
    if !text.text.is_empty() {
        if let Some(timestamp) = agent_timestamp_ms {
            let age = chrono::Utc::now()
                .timestamp_millis()
                .saturating_sub(timestamp)
                .max(0) as u64;
            delivery_ages_ms.push(age);
        }
        *chunks += 1;
        output.push_str(&text.text);
    }
}

fn p99(mut values: Vec<u64>) -> u64 {
    assert!(!values.is_empty(), "fixture must record text delivery ages");
    values.sort_unstable();
    let index = (values.len() * 99).div_ceil(100).saturating_sub(1);
    values[index]
}

async fn connect_and_auth(gateway: GatewaySender) -> acp::ClientSideConnection {
    let agent_config = AgentConfig::default();
    let auth_manager = std::sync::Arc::new(agent_config.create_auth_manager());
    let agent =
        MvpAgent::new(gateway, &agent_config, auth_manager, None).expect("valid agent config");

    let (client_to_agent, agent_from_client) = tokio::io::duplex(DUPLEX_BUFFER_BYTES);
    let (agent_to_client, client_from_agent) = tokio::io::duplex(DUPLEX_BUFFER_BYTES);
    let agent_incoming = LineBufferedRead::spawn_local(agent_from_client.compat());
    let (_agent_conn, agent_io) = acp::AgentSideConnection::new(
        agent,
        agent_to_client.compat_write(),
        agent_incoming,
        |future| {
            tokio::task::spawn_local(future);
        },
    );
    tokio::task::spawn_local(agent_io);

    let client_incoming = LineBufferedRead::spawn_local(client_from_agent.compat());
    let (client_conn, client_io) = acp::ClientSideConnection::new(
        NoopClient,
        client_to_agent.compat_write(),
        client_incoming,
        |future| {
            tokio::task::spawn_local(future);
        },
    );
    tokio::task::spawn_local(client_io);

    let init = client_conn
        .initialize(
            acp::InitializeRequest::new(acp::ProtocolVersion::V1)
                .client_capabilities(
                    acp::ClientCapabilities::new()
                        .fs(acp::FileSystemCapabilities::new())
                        .terminal(false),
                )
                .meta(
                    json!({
                        "startupHints": {
                            "nonInteractive": true,
                            "skipGitStatus": true,
                            "skipProjectLayout": true,
                        },
                        "clientType": "streaming-backpressure-perf",
                        "clientVersion": "0.0-test",
                    })
                    .as_object()
                    .cloned(),
                ),
        )
        .await
        .expect("initialize failed");
    let method = init
        .auth_methods
        .iter()
        .find(|method| &*method.id().0 == "xai.api_key")
        .expect("xai.api_key auth method not advertised");
    client_conn
        .authenticate(
            acp::AuthenticateRequest::new(method.id().clone())
                .meta(json!({ "headless": true }).as_object().cloned()),
        )
        .await
        .expect("authenticate failed");

    client_conn
}

#[test]
fn stalled_acp_streaming_reports_backlog_and_preserves_output() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mock_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("mock runtime");
    let server = mock_runtime
        .block_on(MockInferenceServer::start())
        .expect("mock inference server");
    let workload = Workload::from_env();
    let expected = streamed_text(&workload);
    server.set_response(expected.clone());

    let grok_home = TempDir::new().expect("grok home");
    let workdir = TempDir::new().expect("workdir");
    // This test binary contains only this test. The mock runtime services HTTP
    // only, so it cannot observe these process-level configuration variables.
    unsafe {
        std::env::set_var("GROK_HOME", grok_home.path());
        std::env::set_var("GROK_CLI_CHAT_PROXY_BASE_URL", server.url());
        std::env::set_var("GROK_XAI_API_BASE_URL", server.url());
        std::env::set_var("XAI_API_KEY", "test-key-for-ci");
        std::env::set_var("GROK_TELEMETRY_ENABLED", "false");
        std::env::set_var("GROK_FEEDBACK_ENABLED", "false");
        std::env::set_var("GROK_TRACE_UPLOAD", "false");
    }

    let agent_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("agent runtime");
    let local = tokio::task::LocalSet::new();
    agent_runtime.block_on(local.run_until(async move {
        let (gateway_tx, mut gateway_rx) = tokio::sync::mpsc::unbounded_channel();
        let client_conn = connect_and_auth(GatewaySender::new(gateway_tx)).await;
        let session = client_conn
            .new_session(
                acp::NewSessionRequest::new(workdir.path().to_path_buf())
                    .meta(json!({ "modelId": "test-model" }).as_object().cloned()),
            )
            .await
            .expect("session/new failed");

        // Exclude session-start notifications from the measured turn.
        while gateway_rx.try_recv().is_ok() {}

        let started = Instant::now();
        let mut prompt = Box::pin(client_conn.prompt(acp::PromptRequest::new(
            session.session_id,
            vec![acp::ContentBlock::Text(acp::TextContent::new(
                "stream the fixed backpressure fixture".to_owned(),
            ))],
        )));
        let mut prompt_result = None;
        let mut peak_backlog = 0usize;
        let mut delivered = String::new();
        let mut delivered_chunks = 0usize;
        let mut delivery_ages_ms = Vec::new();

        // This is a sustained producer-over-consumer workload: each received
        // gateway message costs one fixed consumer interval. The source model
        // streams independently, so an uncapped in-process path accumulates
        // both depth and enqueue-to-delivery age.
        while prompt_result.is_none() || delivered.len() < expected.len() {
            tokio::select! {
                result = &mut prompt, if prompt_result.is_none() => {
                    prompt_result = Some(result);
                }
                message = gateway_rx.recv() => {
                    let message = message.expect("agent gateway must remain open during prompt");
                    take_agent_text(
                        message,
                        &mut delivered,
                        &mut delivered_chunks,
                        &mut delivery_ages_ms,
                    );
                    peak_backlog = peak_backlog.max(gateway_rx.len());
                    tokio::time::sleep(workload.consumer_delay).await;
                }
            }
        }

        let prompt_response = prompt_result
            .expect("prompt must resolve after the gateway drains")
            .expect("prompt failed");
        assert!(
            matches!(prompt_response.stop_reason, acp::StopReason::EndTurn),
            "expected terminal EndTurn, got {:?}",
            prompt_response.stop_reason
        );
        assert_eq!(
            delivered, expected,
            "a slow ACP consumer must still receive every streamed byte in order"
        );
        assert!(
            delivered_chunks > 0,
            "fixture must deliver streaming chunks"
        );
        assert_eq!(
            delivery_ages_ms.len(),
            delivered_chunks,
            "every text chunk must retain its production enqueue timestamp"
        );
        let p99_delivery_age_ms = p99(delivery_ages_ms);

        println!(
            "{}",
            json!({
                "metric": "p99_acp_text_delivery_age_ms",
                "value": p99_delivery_age_ms,
            })
        );
        println!(
            "{}",
            json!({
                "metric": "peak_acp_streaming_backlog_messages",
                "value": peak_backlog,
            })
        );
        println!(
            "{}",
            json!({
                "metric": "delivered_stream_chunks",
                "value": delivered_chunks,
            })
        );
        println!(
            "{}",
            json!({
                "metric": "stream_completion_ms",
                "value": started.elapsed().as_secs_f64() * 1_000.0,
            })
        );
    }));
}
