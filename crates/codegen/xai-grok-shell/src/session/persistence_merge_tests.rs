//! Focused regression coverage for streamed ACP text merging.

use super::*;
use std::sync::Arc;

fn test_persistence() -> SessionPersistence {
    let info = Info {
        id: acp::SessionId::new("persistence-merge-test"),
        cwd: "/persistence-merge-test".into(),
    };
    let (tx, rx) = mpsc::unbounded_channel();
    let sampling_client = OaiCompatClient::new(xai_grok_sampler::SamplerConfig::default())
        .expect("create sampling client for persistence fixture");

    SessionPersistence {
        info,
        storage: Arc::new(JsonlStorageAdapter::with_root("/unused".into())),
        pending_notification: None,
        rx,
        remote_sync: None,
        relay_sync: None,
        summary: crate::session::summary::SummaryGenerator::new(
            crate::session::summary::SummaryConfig {
                sampling_client,
                model: String::new(),
                persistence_tx: tx,
            },
        ),
        registry_title_sync: None,
        gateway: None,
    }
}

fn message_notification(session_id: &str, text: impl Into<String>) -> acp::SessionNotification {
    acp::SessionNotification::new(
        acp::SessionId::new(session_id),
        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
            acp::TextContent::new(text.into()),
        ))),
    )
}

fn thought_notification(session_id: &str, text: impl Into<String>) -> acp::SessionNotification {
    acp::SessionNotification::new(
        acp::SessionId::new(session_id),
        acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
            acp::TextContent::new(text.into()),
        ))),
    )
}

fn assert_same_notification(
    actual: &acp::SessionNotification,
    expected: &acp::SessionNotification,
) {
    assert_eq!(
        serde_json::to_value(actual).expect("serialize actual notification"),
        serde_json::to_value(expected).expect("serialize expected notification"),
    );
}

#[test]
fn merge_notification_preserves_stream_and_boundary_semantics() {
    let mut persistence = test_persistence();
    let first = message_notification("first-session", "hello ")
        .meta(serde_json::json!({"eventId": "first"}).as_object().cloned());
    assert!(persistence.maybe_merge_notification(&first).is_none());

    let second = message_notification("current-session", "world").meta(
        serde_json::json!({"eventId": "current"})
            .as_object()
            .cloned(),
    );
    assert!(persistence.maybe_merge_notification(&second).is_none());
    let expected_merged = message_notification("current-session", "hello world").meta(
        serde_json::json!({"eventId": "current"})
            .as_object()
            .cloned(),
    );
    assert_same_notification(
        persistence
            .pending_notification
            .as_ref()
            .expect("merged notification remains pending"),
        &expected_merged,
    );

    let boundary = thought_notification("boundary-session", "reasoning");
    let written = persistence
        .maybe_merge_notification(&boundary)
        .expect("different update kind writes the previous pending notification");
    assert_same_notification(&written, &expected_merged);
    assert_same_notification(
        persistence
            .pending_notification
            .as_ref()
            .expect("boundary update becomes pending"),
        &boundary,
    );

    let thought_follow_up = thought_notification("thought-current", " continues").meta(
        serde_json::json!({"eventId": "thought-current"})
            .as_object()
            .cloned(),
    );
    assert!(
        persistence
            .maybe_merge_notification(&thought_follow_up)
            .is_none()
    );
    let expected_thought = thought_notification("thought-current", "reasoning continues").meta(
        serde_json::json!({"eventId": "thought-current"})
            .as_object()
            .cloned(),
    );
    assert_same_notification(
        persistence
            .pending_notification
            .as_ref()
            .expect("merged thought remains pending"),
        &expected_thought,
    );
}
