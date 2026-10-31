//! Module for types used in the API.
use std::pin::Pin;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use crate::error::AnthropicError;
use crate::DEFAULT_MODEL;

#[derive(Clone, Serialize, Default, Debug, Builder, PartialEq)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "AnthropicError"))]
pub struct CompleteRequest {
    /// The prompt to complete.
    pub prompt: String,
    /// The model to use.
    #[builder(default = "DEFAULT_MODEL.to_string()")]
    pub model: String,
    /// The number of tokens to sample.
    pub max_tokens_to_sample: usize,
    /// The stop sequences to use.
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to incrementally stream the response.
    #[builder(default = "false")]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct CompleteResponse {
    pub completion: String,
    pub stop_reason: Option<StopReason>,
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[default]
    User,
    Assistant,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 {
        media_type: String,
        data: String,
    },
    #[serde(rename = "url")]
    Url {
        url: String,
    },
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentBlock {
    Text {
        text: String,
        /// Sources cited by this text (e.g. web search results), kept as raw JSON so they
        /// round-trip.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<serde_json::Value>>,
    },
    Image {
        source: ImageSource,
    },
    /// Any other block (`thinking`, `server_tool_use`, `web_search_tool_result`, ...), kept
    /// verbatim so it can be sent back unchanged, e.g. to continue a `pause_turn` response.
    #[serde(untagged)]
    Other(serde_json::Value),
}

#[derive(Clone, Serialize, Deserialize, Debug, Builder, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Clone, Serialize, Default, Debug, Builder, PartialEq)]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "AnthropicError"))]
pub struct MessagesRequest {
    /// The User/Assistent prompts.
    pub messages: Vec<Message>,
    /// The System prompt.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub system: String,
    /// The model to use.
    #[builder(default = "DEFAULT_MODEL.to_string()")]
    pub model: String,
    /// The maximum number of tokens to generate before stopping.
    pub max_tokens: usize,
    /// The stop sequences to use.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Whether to incrementally stream the response.
    #[builder(default = "false")]
    pub stream: bool,
    /// Amount of randomness injected into the response.
    ///
    /// Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical /
    /// multiple choice, and closer to 1.0 for creative and generative tasks. Note that even
    /// with temperature of 0.0, the results will not be fully deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Use nucleus sampling.
    ///
    /// In nucleus sampling, we compute the cumulative distribution over all the options for each
    /// subsequent token in decreasing probability order and cut it off once it reaches a particular
    /// probability specified by top_p. You should either alter temperature or top_p, but not both.
    /// Recommended for advanced use cases only. You usually only need to use temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Only sample from the top K options for each subsequent token.
    /// Used to remove "long tail" low probability responses. Learn more technical details here.
    /// Recommended for advanced use cases only. You usually only need to use temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Tools the model may use, e.g. server tools like web search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// Reason for stopping the response generation.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model reached a natural stopping point.
    EndTurn,
    /// The requested max_tokens or the model's maximum was exceeded.
    MaxTokens,
    /// One of the provided custom stop_sequences was generated.
    StopSequence,
    /// The model invoked one or more client tools.
    ToolUse,
    /// A long-running turn with server tools was paused; send the response back as-is to continue.
    PauseTurn,
    /// The model declined to respond.
    Refusal,
    /// The model's context window was exhausted.
    ModelContextWindowExceeded,
    /// A stop reason this crate doesn't know about yet.
    #[serde(other)]
    Unknown,
}

/// Billing and rate-limit usage.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    /// The number of input tokens which were used.
    pub input_tokens: usize,

    /// The number of output tokens which were used.
    pub output_tokens: usize,
}

/// Response body for the Messages API.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Serialize)]
pub struct MessagesResponse {
    /// Unique object identifier.
    /// The format and length of IDs may change over time.
    pub id: String,
    /// Object type.
    /// For Messages, this is always "message".
    pub r#type: String,
    /// Conversational role of the generated message.
    /// This will always be "assistant".
    pub role: Role,
    /// Content generated by the model.
    /// This is an array of content blocks, each of which has a type that determines its shape.
    /// Currently, the only type in responses is "text".
    ///
    /// Example:
    /// `[{"type": "text", "text": "Hi, I'm Claude."}]`
    ///
    /// If the request input messages ended with an assistant turn, then the response content
    /// will continue directly from that last turn. You can use this to constrain the model's
    /// output.
    ///
    /// For example, if the input messages were:
    /// `[ {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},
    ///    {"role": "assistant", "content": "The best answer is ("} ]`
    ///
    /// Then the response content might be:
    /// `[{"type": "text", "text": "B)"}]`
    pub content: Vec<ContentBlock>,
    /// The model that handled the request.
    pub model: String,
    /// The reason that we stopped.
    /// This may be one the following values:
    /// - "end_turn": the model reached a natural stopping point
    /// - "max_tokens": we exceeded the requested max_tokens or the model's maximum
    /// - "stop_sequence": one of your provided custom stop_sequences was generated
    ///
    /// Note that these values are different than those in /v1/complete, where end_turn and
    /// stop_sequence were not differentiated.
    ///
    /// In non-streaming mode this value is always non-null. In streaming mode, it is null
    /// in the message_start event and non-null otherwise.
    pub stop_reason: Option<StopReason>,
    /// Which custom stop sequence was generated, if any.
    /// This value will be a non-null string if one of your custom stop sequences was generated.
    pub stop_sequence: Option<String>,
    /// Billing and rate-limit usage.
    /// Anthropic's API bills and rate-limits by token counts, as tokens represent the underlying
    /// cost to our systems.
    ///
    /// Under the hood, the API transforms requests into a format suitable for the model. The
    /// model's output then goes through a parsing stage before becoming an API response. As a
    /// result, the token counts in usage will not match one-to-one with the exact visible
    /// content of an API request or response.
    ///
    /// For example, output_tokens will be non-zero, even for an empty string response from Claude.
    pub usage: Usage,
}

/// Parsed server side events stream until a [StopReason::StopSequence] is received from server.
pub type CompleteResponseStream = Pin<Box<dyn Stream<Item = Result<CompleteResponse, AnthropicError>> + Send>>;

/// Parsed server side events stream until a [StopReason::StopSequence] is received from server.
pub type MessagesResponseStream = Pin<Box<dyn Stream<Item = Result<MessagesStreamEvent, AnthropicError>> + Send>>;

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
}

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct MessageDeltaUsage {
    pub output_tokens: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct MessageDelta {
    /// The reason that we stopped.
    /// This may be one the following values:
    /// - "end_turn": the model reached a natural stopping point
    /// - "max_tokens": we exceeded the requested max_tokens or the model's maximum
    /// - "stop_sequence": one of your provided custom stop_sequences was generated
    ///
    /// Note that these values are different than those in /v1/complete, where end_turn and
    /// stop_sequence were not differentiated.
    ///
    /// In non-streaming mode this value is always non-null. In streaming mode, it is null
    /// in the message_start event and non-null otherwise.
    pub stop_reason: Option<StopReason>,
    /// Which custom stop sequence was generated, if any.
    /// This value will be a non-null string if one of your custom stop sequences was generated.
    pub stop_sequence: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum MessagesStreamEvent {
    MessageStart { message: Message },
    ContentBlockStart { index: usize, content_block: ContentBlock },
    ContentBlockDelta { index: usize, delta: ContentBlockDelta },
    ContentBlockStop { index: usize },
    MessageDelta { delta: MessageDelta, usage: MessageDeltaUsage },
    MessageStop,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Serialize)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Error ({}): {}", self.error_type, self.message))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn server_tool_blocks_parse_and_round_trip_verbatim() {
        let content = json!([
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Let me search."},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "rust"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                {"type": "web_search_result", "url": "https://www.rust-lang.org/", "title": "Rust",
                 "encrypted_content": "abc", "page_age": null}
            ]},
            {"type": "text", "text": "Rust is a language.", "citations": [
                {"type": "web_search_result_location", "url": "https://www.rust-lang.org/", "title": "Rust",
                 "encrypted_index": "def", "cited_text": "Rust is a language"}
            ]}
        ]);
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": "claude-sonnet-5-5",
            "stop_reason": "pause_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1, "server_tool_use": {"web_search_requests": 1}}
        }))
        .unwrap();

        assert_eq!(response.stop_reason, Some(StopReason::PauseTurn));
        assert_eq!(response.content[1], ContentBlock::Text { text: "Let me search.".into(), citations: None });
        assert!(matches!(&response.content[2], ContentBlock::Other(block) if block["type"] == "server_tool_use"));
        assert!(matches!(&response.content[4], ContentBlock::Text { citations: Some(c), .. } if c.len() == 1));

        // Continuing a paused turn requires sending the blocks back unchanged.
        assert_eq!(serde_json::to_value(&response.content).unwrap(), content);
    }

    #[test]
    fn tools_are_only_sent_when_set() {
        let request = MessagesRequestBuilder::default().build().unwrap();
        assert!(serde_json::to_value(&request).unwrap().get("tools").is_none());

        let tool = json!({"type": "web_search_20250305", "name": "web_search"});
        let request = MessagesRequestBuilder::default().tools(vec![tool.clone()]).build().unwrap();
        assert_eq!(serde_json::to_value(&request).unwrap()["tools"], json!([tool]));
    }

    #[test]
    fn new_and_unknown_stop_reasons_parse() {
        for (raw, expected) in [
            ("tool_use", StopReason::ToolUse),
            ("pause_turn", StopReason::PauseTurn),
            ("refusal", StopReason::Refusal),
            ("model_context_window_exceeded", StopReason::ModelContextWindowExceeded),
            ("some_future_reason", StopReason::Unknown),
        ] {
            assert_eq!(serde_json::from_value::<StopReason>(json!(raw)).unwrap(), expected);
        }
    }
}
