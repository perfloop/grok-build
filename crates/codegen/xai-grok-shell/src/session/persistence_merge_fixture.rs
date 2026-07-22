//! Criterion-only fixture for the consecutive ACP text merge path.
//!
//! This module is feature-gated so production code does not expose the internal
//! persistence actor solely for benchmarking.

use super::{JsonlStorageAdapter, OaiCompatClient, SessionPersistence};
use crate::session::info::Info;
use agent_client_protocol as acp;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Number of mergeable notifications in the fixed long-stream workload.
pub const STREAM_CHUNKS: usize = 128;
/// Byte length of each runtime-built text notification in the workload.
pub const CHUNK_BYTES: usize = 1024;

/// Reusable state for a consecutive, mergeable ACP text stream.
pub struct MergeStreamFixture {
    persistence: SessionPersistence,
    inputs: Vec<acp::SessionNotification>,
    expected: String,
}

impl MergeStreamFixture {
    /// Builds the inputs before measurement so the benchmark isolates the
    /// persistence actor's merge work rather than payload generation.
    pub fn new() -> Self {
        let inputs = (0..STREAM_CHUNKS)
            .map(|index| notification(payload(index)))
            .collect::<Vec<_>>();
        let expected = inputs.iter().map(text).collect::<String>();

        Self {
            persistence: test_persistence(),
            inputs,
            expected,
        }
    }

    /// Merges one complete stream and consumes the resulting text through the
    /// assertion before returning its length to Criterion's black box.
    pub fn merge_stream(&mut self) -> usize {
        self.clear_pending();
        for incoming in &self.inputs {
            assert!(
                self.persistence
                    .maybe_merge_notification(incoming)
                    .is_none()
            );
        }

        let merged = self
            .persistence
            .pending_notification
            .as_ref()
            .expect("merge stream leaves one pending notification");
        assert_eq!(text(merged), self.expected);
        text(merged).len()
    }

    /// Releases the stream result so a Dhat measurement can finish with no
    /// allocation owned by the measured operation.
    pub fn clear_pending(&mut self) {
        self.persistence.pending_notification = None;
    }
}

fn test_persistence() -> SessionPersistence {
    let info = Info {
        id: acp::SessionId::new("persistence-merge-benchmark"),
        cwd: "/persistence-merge-benchmark".into(),
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

fn notification(text: String) -> acp::SessionNotification {
    acp::SessionNotification::new(
        acp::SessionId::new("allocation-stream"),
        acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
            acp::TextContent::new(text),
        ))),
    )
}

fn payload(index: usize) -> String {
    let mut text = format!("chunk-{index:04}:");
    while text.len() < CHUNK_BYTES {
        let byte = b'a' + ((index + text.len()) % 26) as u8;
        text.push(byte as char);
    }
    text
}

fn text(notification: &acp::SessionNotification) -> &str {
    let acp::SessionUpdate::AgentMessageChunk(chunk) = &notification.update else {
        panic!("benchmark inputs must be AgentMessageChunk notifications");
    };
    let acp::ContentBlock::Text(content) = &chunk.content else {
        panic!("benchmark inputs must contain text");
    };
    &content.text
}
