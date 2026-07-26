//! Prompt-entry guard for the session-to-sampler request-handoff proof.
//!
//! This intentionally drives `MvpAgent::prompt` with a live session mailbox.
//! The mailbox is a focused control twin: it verifies the entry point's prompt
//! intake and completion contract without adding unrelated inference work to
//! the allocation selector.

use super::*;
use agent_client_protocol::{self as acp, Agent as _};
use crate::session::{SessionCommand, ok_end_turn};

const PROMPT_GUARD_METRIC: &str = "mvp_agent_prompt_completed";

#[tokio::test(flavor = "current_thread")]
async fn mvp_agent_prompt_routes_live_session() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let agent = build_minimal_agent_for_tests();
            let session_id = acp::SessionId::new("request-move-prompt-guard");
            let (handle, _cmd_tx, mut cmd_rx) = make_live_session_handle(&session_id, None);
            agent.sessions.borrow_mut().insert(session_id.clone(), handle);

            let (prompt_seen_tx, prompt_seen_rx) = tokio::sync::oneshot::channel();
            let mailbox = tokio::task::spawn_local(async move {
                let mut prompt_seen_tx = Some(prompt_seen_tx);
                while let Some(command) = cmd_rx.recv().await {
                    match command {
                        SessionCommand::GetCurrentPromptMode { responds_to } => {
                            let _ = responds_to.send(Default::default());
                        }
                        SessionCommand::GetCurrentModel { responds_to } => {
                            let _ = responds_to.send("request-move-model".to_string());
                        }
                        SessionCommand::Prompt {
                            prompt_blocks,
                            respond_to,
                            ..
                        } => {
                            assert_eq!(prompt_blocks.len(), 1, "prompt reaches the session mailbox");
                            prompt_seen_tx
                                .take()
                                .expect("prompt is dispatched once")
                                .send(())
                                .expect("test waits for the dispatched prompt");
                            let _ = respond_to.send(ok_end_turn(0, None));
                            return;
                        }
                        _ => {}
                    }
                }
                panic!("MvpAgent::prompt closed the mailbox without dispatching a prompt");
            });

            agent
                .prompt(acp::PromptRequest::new(
                    session_id,
                    vec![acp::ContentBlock::from("request-move-prompt-guard")],
                ))
                .await
                .expect("MvpAgent::prompt completes after the session response");
            prompt_seen_rx
                .await
                .expect("MvpAgent::prompt dispatches the prompt to the live session");
            mailbox.await.expect("prompt mailbox task does not panic");

            if let Ok(result_path) = std::env::var("PERFLOOP_RESULT_FILE") {
                std::fs::write(
                    result_path,
                    format!(
                        "{}\n",
                        serde_json::json!({
                            "metric": PROMPT_GUARD_METRIC,
                            "value": 1,
                            "unit": "completed_prompt",
                        })
                    ),
                )
                .expect("write Perfloop prompt-entry guard sample");
            }
        })
        .await;
}
