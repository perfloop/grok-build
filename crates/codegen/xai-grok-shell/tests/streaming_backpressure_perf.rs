//! End-to-end stalled-ACP workload for the streaming backpressure path.
//!
//! A loopback model emits a long Chat Completions response while the agent's
//! outbound ACP gateway is deliberately left undrained.  The workload uses the
//! real `MvpAgent -> SessionActor -> SamplerActor -> drive_l2` path; after the
//! fixed stall it drains the gateway and proves every streamed byte and the
//! terminal prompt response survive in order.
//!
//! Run:
//!   cargo test --release -p xai-grok-shell --test streaming_backpressure_perf -- --exact stalled_acp_streaming_reports_backlog_and_preserves_output --nocapture

use std::time::{Duration, Instant};

use agent_client_protocol::{self as acp, Agent as _};
use serde_json::json;
use tempfile::TempDir;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use xai_acp_lib::{AcpAgentGatewaySender as GatewaySender, AcpClientMessage, LineBufferedRead};
use xai_grok_shell::agent::config::Config as AgentConfig;
use xai_grok_shell::agent::mvp_agent::MvpAgent;
use xai_grok_test_support::MockInferenceServer;

const STREAM_CHUNKS: usize = 1_024;
const ACP_STALL: Duration = Duration::from_millis(150);
const COMPLETE_TIMEOUT: Duration = Duration::from_secs(20);
const DUPLEX_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// The normal client-side RPC dispatcher is intentionally not installed for
/// this workload.  The agent still receives initialize/new-session/prompt
/// requests over the duplex connection, while its real outbound gateway queue
/// remains stalled for `ACP_STALL`.
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

fn streamed_text() -> String {
    (0..STREAM_CHUNKS)
        .map(|index| format!("chunk-{index:04}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn take_agent_text(message: AcpClientMessage, output: &mut String, chunks: &mut usize) {
    let AcpClientMessage::SessionNotification(args) = message else {
        return;
    };
    let acp::SessionUpdate::AgentMessageChunk(chunk) = args.request.update else {
        return;
    };
    let acp::ContentBlock::Text(text) = chunk.content else {
        return;
    };
    if !text.text.is_empty() {
        *chunks += 1;
        output.push_str(&text.text);
    }
}

async fn connect_and_auth(gateway: GatewaySender) -> acp::ClientSideConnection {
    let agent_config = AgentConfig::default();
    let auth_manager = std::sync::Arc::new(agent_config.create_auth_manager());
    let agent =
        MvpAgent::new(gateway, &agent_config, auth_manager, None).expect("valid agent config");

    let (client_to_agent, agent_from_client) = tokio::io::duplex(DUPLEX_BUFFER_BYTES);
    let (agent_to_client, client_from_agent) = tokio::io::duplex(DUPLEX_BUFFER_BYTES);
    let agent_incoming = LineBufferedRead::spawn_local(agent_from_client.compat());
    let (agent_conn, agent_io) = acp::AgentSideConnection::new(
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

    let init = tokio::time::timeout(
        COMPLETE_TIMEOUT,
        client_conn.initialize(
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
        ),
    )
    .await
    .expect("initialize timed out")
    .expect("initialize failed");
    let method = init
        .auth_methods
        .iter()
        .find(|method| &*method.id().0 == "xai.api_key")
        .expect("xai.api_key auth method not advertised");
    tokio::time::timeout(
        COMPLETE_TIMEOUT,
        client_conn.authenticate(
            acp::AuthenticateRequest::new(method.id().clone())
                .meta(json!({ "headless": true }).as_object().cloned()),
        ),
    )
    .await
    .expect("authenticate timed out")
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
    let expected = streamed_text();
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
        let session = tokio::time::timeout(
            COMPLETE_TIMEOUT,
            client_conn.new_session(
                acp::NewSessionRequest::new(workdir.path().to_path_buf())
                    .meta(json!({ "modelId": "test-model" }).as_object().cloned()),
            ),
        )
        .await
        .expect("session/new timed out")
        .expect("session/new failed");

        // Exclude session-start notifications from the deliberate prompt stall.
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
        let stall_deadline = tokio::time::Instant::now() + ACP_STALL;

        // Poll the prompt while leaving the actual outbound gateway receiver
        // untouched. The queue length is therefore the real queued ACP work at
        // the product delivery boundary, not a synthetic counter.
        while tokio::time::Instant::now() < stall_deadline {
            tokio::select! {
                result = &mut prompt, if prompt_result.is_none() => {
                    prompt_result = Some(result);
                }
                _ = tokio::time::sleep(Duration::from_millis(1)) => {}
            }
            peak_backlog = peak_backlog.max(gateway_rx.len());
        }

        let mut delivered = String::new();
        let mut delivered_chunks = 0usize;
        let drain_deadline = tokio::time::Instant::now() + COMPLETE_TIMEOUT;
        while prompt_result.is_none() || delivered.len() < expected.len() {
            tokio::select! {
                result = &mut prompt, if prompt_result.is_none() => {
                    prompt_result = Some(result);
                }
                message = gateway_rx.recv() => {
                    let message = message.expect("agent gateway must remain open during prompt");
                    take_agent_text(message, &mut delivered, &mut delivered_chunks);
                }
                _ = tokio::time::sleep_until(drain_deadline) => {
                    panic!(
                        "stream did not drain within {:?}: delivered {} of {} bytes",
                        COMPLETE_TIMEOUT,
                        delivered.len(),
                        expected.len(),
                    );
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
            "a stalled ACP receiver must still receive every streamed byte in order"
        );
        assert!(
            delivered_chunks > 0,
            "fixture must deliver streaming chunks"
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
