//! Anthropic ingress adapter

use crate::types::{IngressError, IngressResult};
use axum::{
    Router,
    extract::{Json, RawQuery, State},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::post,
};
use futures::StreamExt;
#[cfg(test)]
use lunaroute_core::normalized::{Choice, Usage};
use lunaroute_core::{
    normalized::{
        ContentPart, FinishReason, FunctionCall, FunctionDefinition, Message, MessageContent,
        NormalizedRequest, NormalizedResponse, NormalizedStreamEvent, Role, Tool, ToolCall,
    },
    provider::Provider,
    session_store::SessionStore,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Maximum size for tool arguments (1MB)
const MAX_TOOL_ARGS_SIZE: usize = 1_000_000;

/// Extract a session ID from `metadata.user_id`. Handles three shapes the
/// Anthropic client has used:
///   1. Object: `{"session_id": "...", ...}`
///   2. JSON-encoded string: `"{\"session_id\":\"...\"}"`
///   3. Legacy flat: `"user_<hash>_account_<uuid>_session_<uuid>"`
///
/// Returns `None` if no shape matches or the candidate fails validation
/// (length / charset). Validation enforces alphanumeric, dash, underscore
/// only to prevent path traversal in downstream session-id consumers.
fn extract_session_id_from_user_id(value: &serde_json::Value) -> Option<String> {
    // Shape 1: object literal
    if let Some(obj) = value.as_object() {
        let candidate = obj.get("session_id").and_then(|v| v.as_str())?;
        return validate_session_id_candidate(candidate);
    }

    let s = value.as_str()?;

    // Shape 2: JSON-encoded string (current Claude clients send this)
    if s.trim_start().starts_with('{') {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::from_str::<serde_json::Value>(s)
            && let Some(candidate) = obj.get("session_id").and_then(|v| v.as_str())
        {
            return validate_session_id_candidate(candidate);
        }
        // JSON-shaped but unparseable or missing session_id; nothing to recover.
        tracing::debug!("metadata.user_id looked like JSON but had no usable session_id");
        return None;
    }

    // Shape 3: legacy `user_<hash>_account_<uuid>_session_<uuid>` or bare token
    let candidate = if let Some(pos) = s.rfind("_session_") {
        &s[pos + "_session_".len()..]
    } else {
        s
    };
    validate_session_id_candidate(candidate)
}

fn validate_session_id_candidate(candidate: &str) -> Option<String> {
    if candidate.is_empty() {
        tracing::debug!("Empty session ID extracted from user_id");
        return None;
    }
    if candidate.len() > 255 {
        tracing::warn!("Session ID too long: {} chars (max 255)", candidate.len());
        return None;
    }
    if !candidate
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        tracing::warn!(
            "Invalid session ID format (contains unsafe characters): {}",
            candidate
        );
        return None;
    }
    Some(candidate.to_string())
}

/// Validate tool input schema (must be valid JSON Schema)
fn validate_tool_schema(schema: &serde_json::Value, tool_name: &str) -> IngressResult<()> {
    // Ensure it's a valid JSON Schema object
    if !schema.is_object() {
        return Err(IngressError::InvalidRequest(format!(
            "Tool '{}': input_schema must be a valid JSON Schema object",
            tool_name
        )));
    }

    // Check for required "type" field (common JSON Schema requirement)
    if schema.get("type").is_none() {
        return Err(IngressError::InvalidRequest(format!(
            "Tool '{}': schema must have 'type' field",
            tool_name
        )));
    }

    Ok(())
}

/// Anthropic system parameter (can be string or array of blocks)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

/// Anthropic system content block
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicSystemBlock {
    Text {
        text: String,
    },
    #[serde(skip)]
    Unknown(serde_json::Value),
}

impl<'de> serde::Deserialize<'de> for AnthropicSystemBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let block_type = value.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            "text" => {
                let text = value
                    .get("text")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("text"))?
                    .to_string();
                Ok(Self::Text { text })
            }
            _ => Ok(Self::Unknown(value)),
        }
    }
}

/// Anthropic messages request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
}

/// Anthropic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<AnthropicMessageContent>,
}

/// Anthropic message content (text string or array of content blocks)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic content block (for requests)
///
/// Uses custom Deserialize to gracefully handle unknown block types (e.g. `thinking`,
/// `redacted_thinking`, `image`) that Claude Code may send but we don't need to normalize.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: Option<bool>,
    },
    /// Catch-all for unknown block types (thinking, image, etc.) — skipped during normalization
    #[serde(skip)]
    Unknown(serde_json::Value),
}

impl<'de> serde::Deserialize<'de> for AnthropicContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let block_type = value.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            "text" => {
                let text = value
                    .get("text")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("text"))?
                    .to_string();
                Ok(Self::Text { text })
            }
            "tool_use" => {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("id"))?
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?
                    .to_string();
                let input = value
                    .get("input")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                Ok(Self::ToolUse { id, name, input })
            }
            "tool_result" => {
                let tool_use_id = value
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("tool_use_id"))?
                    .to_string();
                // Content can be a string, an array of content blocks, or absent (null/missing)
                let content = match value.get("content") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(arr)) => arr
                        .iter()
                        .filter_map(|b| {
                            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                                b.get("text").and_then(|t| t.as_str()).map(String::from)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Some(serde_json::Value::Null) | None => String::new(),
                    Some(other) => {
                        return Err(serde::de::Error::custom(format!(
                            "tool_result.content must be a string, array, or null; got {}",
                            other
                        )));
                    }
                };
                let is_error = value.get("is_error").and_then(|v| v.as_bool());
                Ok(Self::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                })
            }
            _ => Ok(Self::Unknown(value)),
        }
    }
}

/// Anthropic response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub role: String,
    pub content: Vec<AnthropicContent>,
    pub model: String,
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Anthropic content block (for responses)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Anthropic usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Anthropic tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Anthropic streaming event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: AnthropicContentDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: AnthropicUsageDelta,
    },
    MessageStop,
}

/// Anthropic message start
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub role: String,
    pub model: String,
    pub usage: AnthropicUsage,
}

/// Anthropic content block start
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockStart {
    Text { text: String },
    ToolUse { id: String, name: String },
}

/// Anthropic content delta
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

/// Anthropic message delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Anthropic usage delta (for streaming)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsageDelta {
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,
}

/// Validate Anthropic request parameters
fn validate_request(req: &AnthropicMessagesRequest) -> IngressResult<()> {
    // Validate temperature (0.0 to 1.0 for Anthropic)
    if let Some(temp) = req.temperature
        && !(0.0..=1.0).contains(&temp)
    {
        return Err(IngressError::InvalidRequest(format!(
            "temperature must be between 0.0 and 1.0, got {}",
            temp
        )));
    }

    // Validate top_p (0.0 to 1.0)
    if let Some(top_p) = req.top_p
        && !(0.0..=1.0).contains(&top_p)
    {
        return Err(IngressError::InvalidRequest(format!(
            "top_p must be between 0.0 and 1.0, got {}",
            top_p
        )));
    }

    // Validate top_k (positive integer)
    if let Some(top_k) = req.top_k
        && top_k == 0
    {
        return Err(IngressError::InvalidRequest(
            "top_k must be greater than 0".to_string(),
        ));
    }

    // Validate max_tokens (required and positive for Anthropic)
    if let Some(max_tokens) = req.max_tokens {
        if max_tokens == 0 {
            return Err(IngressError::InvalidRequest(
                "max_tokens must be greater than 0".to_string(),
            ));
        }
        if max_tokens > 100000 {
            return Err(IngressError::InvalidRequest(format!(
                "max_tokens must be <= 100000, got {}",
                max_tokens
            )));
        }
    }

    // Validate model name is not empty and length
    if req.model.is_empty() {
        return Err(IngressError::InvalidRequest(
            "model field cannot be empty".to_string(),
        ));
    }
    if req.model.len() > 256 {
        return Err(IngressError::InvalidRequest(format!(
            "model name too long: {} chars (max 256)",
            req.model.len()
        )));
    }

    // Validate messages array is not empty
    if req.messages.is_empty() {
        return Err(IngressError::InvalidRequest(
            "messages array cannot be empty".to_string(),
        ));
    }

    // Validate messages array length (max 100,000 per Anthropic spec)
    if req.messages.len() > 100_000 {
        return Err(IngressError::InvalidRequest(format!(
            "messages array too large: {} messages (max 100,000)",
            req.messages.len()
        )));
    }

    Ok(())
}

/// Convert Anthropic request to normalized request
pub fn to_normalized(req: AnthropicMessagesRequest) -> IngressResult<NormalizedRequest> {
    use lunaroute_core::normalized::ToolResult;

    // Validate request parameters first
    validate_request(&req)?;

    // Collect both messages and tool results
    let mut tool_results = Vec::new();

    let messages: Result<Vec<Message>, IngressError> = req
        .messages
        .into_iter()
        .map(|msg| {
            let role = match msg.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => {
                    return Err(IngressError::InvalidRequest(format!(
                        "Invalid role: {} (Anthropic only supports 'user' and 'assistant')",
                        msg.role
                    )));
                }
            };

            let content = msg
                .content
                .unwrap_or(AnthropicMessageContent::Text(String::new()));

            // Parse content and extract tool calls
            let (text_content, tool_calls, tool_call_id, _tool_is_error) = match content {
                AnthropicMessageContent::Text(text) => {
                    // Validate message content length
                    if text.len() > 1_000_000 {
                        return Err(IngressError::InvalidRequest(format!(
                            "Message content too large: {} bytes (max 1MB)",
                            text.len()
                        )));
                    }
                    (text, vec![], None, None)
                }
                AnthropicMessageContent::Blocks(blocks) => {
                    let mut text_parts = Vec::new();
                    let mut tool_calls_vec = Vec::new();
                    let mut tool_result_id = None;
                    let mut tool_result_is_error = None;

                    for block in blocks {
                        match block {
                            AnthropicContentBlock::Text { text } => {
                                text_parts.push(text);
                            }
                            AnthropicContentBlock::ToolUse { id, name, input } => {
                                let arguments = serde_json::to_string(&input).map_err(|e| {
                                    tracing::warn!("Failed to serialize tool input: {}", e);
                                    IngressError::InvalidRequest(format!(
                                        "Invalid tool input for '{}': {}",
                                        name, e
                                    ))
                                })?;

                                // Validate tool arguments size
                                if arguments.len() > MAX_TOOL_ARGS_SIZE {
                                    return Err(IngressError::InvalidRequest(format!(
                                        "Tool arguments too large for '{}': {} bytes (max 1MB)",
                                        name,
                                        arguments.len()
                                    )));
                                }

                                tool_calls_vec.push(ToolCall {
                                    id,
                                    tool_type: "function".to_string(),
                                    function: FunctionCall { name, arguments },
                                });
                            }
                            AnthropicContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => {
                                tool_result_id = Some(tool_use_id.clone());
                                tool_result_is_error = is_error;
                                text_parts.push(content.clone());

                                // Add to tool_results collection
                                tool_results.push(ToolResult {
                                    tool_call_id: tool_use_id,
                                    is_error: is_error.unwrap_or(false),
                                    content,
                                    tool_name: None, // Will be filled in by SessionRecorder
                                });
                            }
                            AnthropicContentBlock::Unknown(ref val) => {
                                // Skip unknown block types (thinking, redacted_thinking, image, etc.)
                                tracing::debug!(
                                    "Skipping unknown content block type during normalization: {}",
                                    val.get("type")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("unknown")
                                );
                            }
                        }
                    }

                    (
                        text_parts.join("\n"),
                        tool_calls_vec,
                        tool_result_id,
                        tool_result_is_error,
                    )
                }
            };

            Ok(Message {
                role,
                content: MessageContent::Text(text_content),
                name: None,
                tool_calls,
                tool_call_id,
            })
        })
        .collect();

    // Convert tools with validation
    let tools = if let Some(tools) = req.tools {
        let result: Result<Vec<Tool>, IngressError> = tools
            .into_iter()
            .map(|tool| {
                // Validate tool input schema
                validate_tool_schema(&tool.input_schema, &tool.name)?;

                Ok(Tool {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.input_schema,
                    },
                })
            })
            .collect();
        result?
    } else {
        vec![]
    };

    // Convert system parameter to string (supports both string and array formats)
    let system = req.system.map(|sys| match sys {
        AnthropicSystem::Text(text) => text,
        AnthropicSystem::Blocks(blocks) => {
            // Concatenate all text blocks, skip unknown types
            blocks
                .into_iter()
                .filter_map(|block| match block {
                    AnthropicSystemBlock::Text { text } => Some(text),
                    AnthropicSystemBlock::Unknown(_) => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    });

    Ok(NormalizedRequest {
        messages: messages?,
        system,
        model: req.model,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop_sequences: req.stop_sequences.unwrap_or_default(),
        stream: req.stream.unwrap_or(false),
        tools,
        tool_choice: None, // Anthropic doesn't have tool_choice in same way
        tool_results,      // Tool results extracted from messages
        metadata: std::collections::HashMap::new(),
    })
}

/// Ensure a tool ID is valid for the Anthropic API (`^[a-zA-Z0-9_-]+$`).
/// Converts OpenAI-format IDs (e.g. `call_xxx`) to Anthropic-format (`toolu_call_xxx`)
/// and replaces invalid characters with `_` to preserve position information.
/// When sanitization alters the ID, appends a 4-char hash of the original to
/// prevent collisions between distinct upstream IDs that sanitize to the same form.
/// Avoids double-prefixing IDs that already start with `toolu_`.
fn to_anthropic_tool_id(id: &str) -> String {
    use std::hash::{Hash, Hasher};

    // Replace invalid chars with underscore (preserves structure)
    let sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() || sanitized.chars().all(|c| c == '_') {
        return format!("toolu_{}", uuid::Uuid::new_v4().simple());
    }

    // If sanitization changed the ID, append a short hash to prevent collisions
    let base = if sanitized != id {
        let mut hasher = std::hash::DefaultHasher::new();
        id.hash(&mut hasher);
        format!("{}_{:08x}", sanitized, hasher.finish() as u32)
    } else {
        sanitized
    };

    // If base already has toolu_ prefix, keep it as-is
    if base.starts_with("toolu_") {
        base
    } else {
        format!("toolu_{}", base)
    }
}

/// Convert normalized response to Anthropic response
pub fn from_normalized(resp: NormalizedResponse) -> AnthropicResponse {
    let content: Vec<AnthropicContent> = resp
        .choices
        .first()
        .map(|choice| {
            let mut content_blocks = Vec::new();

            // Add text content if present
            match &choice.message.content {
                MessageContent::Text(text) => {
                    if !text.is_empty() {
                        content_blocks.push(AnthropicContent::Text {
                            text: text.clone(),
                        });
                    }
                },
                MessageContent::Parts(parts) => {
                    // Extract text parts from multimodal content
                    // Anthropic responses with images would need specific handling,
                    // but for now we extract text parts
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => {
                                if !text.is_empty() {
                                    content_blocks.push(AnthropicContent::Text {
                                        text: text.clone(),
                                    });
                                }
                            },
                            ContentPart::Image { .. } => {
                                tracing::warn!("Image content in response not supported for Anthropic format, skipping");
                            }
                        }
                    }
                },
            }

            // Add tool_use blocks
            for tool_call in &choice.message.tool_calls {
                // Parse arguments back to JSON
                let input = match serde_json::from_str(&tool_call.function.arguments) {
                    Ok(value) => value,
                    Err(e) => {
                        tracing::error!(
                            "Failed to parse tool call arguments for '{}': {}. Using empty object.",
                            tool_call.function.name, e
                        );
                        serde_json::Value::Object(serde_json::Map::new())
                    }
                };

                content_blocks.push(AnthropicContent::ToolUse {
                    id: to_anthropic_tool_id(&tool_call.id),
                    name: tool_call.function.name.clone(),
                    input,
                });
            }

            content_blocks
        })
        .unwrap_or_default();

    let stop_reason = resp
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_ref())
        .map(|fr| match fr {
            FinishReason::Stop => "end_turn".to_string(),
            FinishReason::Length => "max_tokens".to_string(),
            FinishReason::ToolCalls => "tool_use".to_string(),
            // ContentFilter doesn't exist in Anthropic spec - map to end_turn
            FinishReason::ContentFilter => "end_turn".to_string(),
            // Error is not in spec but useful for debugging
            FinishReason::Error => "end_turn".to_string(),
        });

    AnthropicResponse {
        // Use ID as-is (egress already has proper format from provider)
        id: resp.id,
        type_: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason,
        stop_sequence: None, // TODO: Track which stop sequence was hit
        usage: AnthropicUsage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    }
}

/// Tracks streaming content block state with unique Anthropic index allocation.
/// OpenAI tool_call_index can collide with text delta indices (both start at 0),
/// so we allocate monotonically increasing Anthropic indices to avoid duplicates.
#[derive(Debug, Clone, Default)]
struct StreamBlockState {
    /// The Anthropic block index of the currently active block, if any
    active_index: Option<u32>,
    /// Whether the active block is a tool block
    active_is_tool: bool,
    /// The OpenAI tool_call_index of the active tool block (for matching deltas)
    active_tool_call_index: Option<u32>,
    /// Next Anthropic content block index to allocate
    next_index: u32,
    /// Maps OpenAI tool_call_index → allocated Anthropic block index
    tool_index_map: std::collections::HashMap<u32, u32>,
}

impl StreamBlockState {
    fn close_active(&mut self, events: &mut Vec<AnthropicStreamEvent>) {
        if let Some(idx) = self.active_index.take() {
            events.push(AnthropicStreamEvent::ContentBlockStop { index: idx });
            self.active_is_tool = false;
            self.active_tool_call_index = None;
        }
    }

    fn alloc_index(&mut self) -> u32 {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }
}

/// Convert normalized stream event to Anthropic SSE events
/// Returns a vector because some normalized events map to multiple Anthropic events
///
/// State Management:
/// - `state` tracks the active block, allocates unique Anthropic indices, and maps
///   OpenAI tool_call_index values to allocated indices to prevent collisions
/// - State is maintained per-stream via scan() combinator (sequential, no concurrency)
fn stream_event_to_anthropic_events(
    event: NormalizedStreamEvent,
    stream_id: &str,
    model: &str,
    state: &mut StreamBlockState,
) -> Vec<AnthropicStreamEvent> {
    use tracing::debug;
    match event {
        NormalizedStreamEvent::Start { .. } => {
            // Reset state for new stream
            *state = StreamBlockState::default();
            debug!("Anthropic stream started: stream_id={}", stream_id);

            // Anthropic starts with message_start event
            vec![AnthropicStreamEvent::MessageStart {
                message: AnthropicMessageStart {
                    id: stream_id.to_string(),
                    type_: "message".to_string(),
                    role: "assistant".to_string(),
                    model: model.to_string(),
                    usage: AnthropicUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    },
                },
            }]
        }
        NormalizedStreamEvent::Delta { index: _, delta } => {
            let mut events = Vec::new();

            // If a non-text block is active, close it first
            if state.active_index.is_some() && state.active_is_tool {
                state.close_active(&mut events);
            }

            // Open text block if not already active
            if state.active_index.is_none() {
                let idx = state.alloc_index();
                events.push(AnthropicStreamEvent::ContentBlockStart {
                    index: idx,
                    content_block: AnthropicContentBlockStart::Text {
                        text: String::new(),
                    },
                });
                state.active_index = Some(idx);
                state.active_is_tool = false;
            }

            // Send the text delta
            if let Some(content) = delta.content {
                events.push(AnthropicStreamEvent::ContentBlockDelta {
                    index: state.active_index.unwrap(),
                    delta: AnthropicContentDelta::TextDelta { text: content },
                });
            }

            events
        }
        NormalizedStreamEvent::ToolCallDelta {
            index: _,
            tool_call_index,
            id,
            function,
        } => {
            let mut events = Vec::new();

            // Start tool use block if we have id and name, and it's not already active
            if let (Some(tool_id), Some(func)) =
                (id, function.as_ref().and_then(|f| f.name.clone()))
            {
                let already_active =
                    state.active_is_tool && state.active_tool_call_index == Some(tool_call_index);

                if !already_active {
                    // Close previous block if one is active
                    state.close_active(&mut events);

                    // Allocate a unique Anthropic index for this tool block
                    let idx = state.alloc_index();
                    state.tool_index_map.insert(tool_call_index, idx);

                    events.push(AnthropicStreamEvent::ContentBlockStart {
                        index: idx,
                        content_block: AnthropicContentBlockStart::ToolUse {
                            id: to_anthropic_tool_id(&tool_id),
                            name: func,
                        },
                    });
                    state.active_index = Some(idx);
                    state.active_is_tool = true;
                    state.active_tool_call_index = Some(tool_call_index);
                }
            }

            // Send partial JSON only if the matching tool block is active
            if let Some(func) = function.and_then(|f| f.arguments) {
                if state.active_is_tool && state.active_tool_call_index == Some(tool_call_index) {
                    events.push(AnthropicStreamEvent::ContentBlockDelta {
                        index: state.active_index.unwrap(),
                        delta: AnthropicContentDelta::InputJsonDelta { partial_json: func },
                    });
                } else {
                    tracing::debug!(
                        "Dropping tool argument delta for block {} — no matching active tool block",
                        tool_call_index
                    );
                }
            }

            events
        }
        NormalizedStreamEvent::Usage { .. } => {
            // Usage is sent with message_delta at the end
            vec![]
        }
        NormalizedStreamEvent::End { finish_reason } => {
            let mut events = Vec::new();

            // Close whichever content block is currently active (text or tool)
            if state.active_index.is_some() {
                state.close_active(&mut events);
                debug!("Anthropic content block closed");
            } else {
                debug!("Anthropic stream ended without content block");
            }

            // Send message_delta with stop_reason
            let stop_reason = match finish_reason {
                FinishReason::Stop => "end_turn",
                FinishReason::Length => "max_tokens",
                FinishReason::ToolCalls => "tool_use",
                FinishReason::ContentFilter => "end_turn",
                FinishReason::Error => "end_turn",
            };

            events.push(AnthropicStreamEvent::MessageDelta {
                delta: AnthropicMessageDelta {
                    stop_reason: Some(stop_reason.to_string()),
                    stop_sequence: None,
                },
                usage: AnthropicUsageDelta {
                    output_tokens: 0,
                    thinking_tokens: None,
                },
            });

            // Send message_stop
            events.push(AnthropicStreamEvent::MessageStop);

            events
        }
        NormalizedStreamEvent::Error { .. } => {
            // Errors are handled separately
            vec![]
        }
    }
}

/// Messages handler
pub async fn messages(
    State(provider): State<Arc<dyn Provider>>,
    Json(req): Json<AnthropicMessagesRequest>,
) -> Result<Response, IngressError> {
    let start_time = std::time::Instant::now();
    let is_streaming = req.stream.unwrap_or(false);
    let model = req.model.clone();

    // Convert to normalized format (includes validation)
    let normalized = to_normalized(req)?;

    if is_streaming {
        // Log streaming request for observability
        tracing::debug!(
            "Anthropic streaming request: model={}, messages={}",
            model,
            normalized.messages.len()
        );

        // Validate that provider supports streaming
        if !provider.capabilities().supports_streaming {
            return Err(IngressError::UnsupportedFeature(
                "Provider does not support streaming".to_string(),
            ));
        }

        // Call provider.stream()
        let stream = provider
            .stream(normalized)
            .await
            .map_err(|e| IngressError::ProviderError(e.to_string()))?;

        // Generate stream ID (Anthropic uses msg_* prefix) and wrap in Arc for efficient sharing
        let stream_id = Arc::new(format!("msg_{}", uuid::Uuid::new_v4().simple()));
        let model = Arc::new(model);

        // Convert normalized events to Anthropic SSE format
        // Use scan to maintain state across stream events
        let sse_stream = stream
            .scan(StreamBlockState::default(), move |block_state, result| {
                let stream_id = Arc::clone(&stream_id);
                let model = Arc::clone(&model);

                let events = match result {
                    Ok(event) => {
                        let anthropic_events = stream_event_to_anthropic_events(
                            event,
                            stream_id.as_str(),
                            model.as_str(),
                            block_state,
                        );

                        anthropic_events
                            .into_iter()
                            .map(|evt| match Event::default().json_data(evt) {
                                Ok(event) => Ok::<_, IngressError>(event),
                                Err(e) => Err(IngressError::Internal(format!(
                                    "Failed to create SSE event: {}",
                                    e
                                ))),
                            })
                            .collect::<Vec<_>>()
                    }
                    Err(e) => {
                        let mut error_events = Vec::new();

                        // Close any active content block before the error
                        if let Some(idx) = block_state.active_index.take()
                            && let Ok(evt) = Event::default()
                                .json_data(AnthropicStreamEvent::ContentBlockStop { index: idx })
                        {
                            error_events.push(Ok(evt));
                        }

                        // Send error event
                        let error_payload = serde_json::json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": e.to_string()
                            }
                        });
                        match Event::default().json_data(error_payload) {
                            Ok(event) => error_events.push(Ok(event)),
                            Err(_) => error_events.push(Err(IngressError::Internal(
                                "Failed to create error SSE event".to_string(),
                            ))),
                        }

                        // Send message_stop to properly terminate the stream
                        if let Ok(evt) =
                            Event::default().json_data(AnthropicStreamEvent::MessageStop)
                        {
                            error_events.push(Ok(evt));
                        }

                        error_events
                    }
                };

                futures::future::ready(Some(events))
            })
            .flat_map(futures::stream::iter);

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let before_provider = std::time::Instant::now();
        let pre_provider_overhead = before_provider.duration_since(start_time);
        tracing::debug!(
            "Proxy overhead before provider call (normalization): {:.2}ms",
            pre_provider_overhead.as_secs_f64() * 1000.0
        );

        // Call provider
        let normalized_response = provider
            .send(normalized)
            .await
            .map_err(|e| IngressError::ProviderError(e.to_string()))?;

        let after_provider = std::time::Instant::now();
        let provider_time = after_provider.duration_since(before_provider);
        tracing::debug!(
            "Provider response time: {:.2}ms",
            provider_time.as_secs_f64() * 1000.0
        );

        // Convert back to Anthropic format
        let response = from_normalized(normalized_response);

        let response_result = Ok(Json(response).into_response());

        let total_time = std::time::Instant::now().duration_since(start_time);
        let post_provider_overhead = total_time - provider_time - pre_provider_overhead;
        tracing::debug!(
            "Proxy overhead after provider response (denormalization): {:.2}ms",
            post_provider_overhead.as_secs_f64() * 1000.0
        );
        tracing::debug!(
            "Total proxy overhead: {:.2}ms (pre: {:.2}ms + post: {:.2}ms), provider: {:.2}ms, total: {:.2}ms",
            (pre_provider_overhead + post_provider_overhead).as_secs_f64() * 1000.0,
            pre_provider_overhead.as_secs_f64() * 1000.0,
            post_provider_overhead.as_secs_f64() * 1000.0,
            provider_time.as_secs_f64() * 1000.0,
            total_time.as_secs_f64() * 1000.0
        );

        response_result
    }
}

/// State for passthrough handler (connector + optional stats tracker + metrics + session store)
pub struct PassthroughState {
    pub connector: Arc<lunaroute_egress::anthropic::AnthropicConnector>,
    pub stats_tracker: Option<Arc<dyn crate::types::SessionStatsTracker>>,
    pub metrics: Option<Arc<lunaroute_observability::Metrics>>,
    pub session_store: Option<Arc<dyn SessionStore>>,
    pub tool_call_mapper: Arc<lunaroute_session::ToolCallMapper>,
    pub sse_keepalive_interval_secs: u64,
    pub sse_keepalive_enabled: bool,
    pub provider_registry: Option<Arc<crate::ProviderRegistry>>,
}

#[derive(Default)]
struct PassthroughStreamStatus {
    message_stopped: bool,
    error: Option<String>,
}

/// Headers for a new HTTP connection and body framing chosen by the server.
fn forwardable_headers(headers: &axum::http::HeaderMap) -> axum::http::HeaderMap {
    let mut forwarded = headers.clone();
    for value in headers.get_all(axum::http::header::CONNECTION) {
        if let Ok(value) = value.to_str() {
            for name in value.split(',') {
                if let Ok(name) = axum::http::HeaderName::from_bytes(name.trim().as_bytes()) {
                    forwarded.remove(name);
                }
            }
        }
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    ] {
        forwarded.remove(name);
    }
    forwarded
}

async fn record_passthrough_error(
    state: &PassthroughState,
    session_id: &Option<String>,
    request_id: &Option<String>,
    started: std::time::Instant,
    error: String,
) {
    if let (Some(store), Some(session_id), Some(request_id)) =
        (&state.session_store, session_id, request_id)
    {
        let event = lunaroute_session::SessionEvent::Completed {
            session_id: session_id.clone(),
            request_id: request_id.clone(),
            timestamp: chrono::Utc::now(),
            success: false,
            error: Some(error),
            finish_reason: None,
            final_stats: Box::new(lunaroute_session::events::FinalSessionStats {
                total_duration_ms: started.elapsed().as_millis() as u64,
                provider_time_ms: 0,
                proxy_overhead_ms: 0.0,
                total_tokens: Default::default(),
                tool_summary: Default::default(),
                performance: Default::default(),
                streaming_stats: None,
                estimated_cost: None,
            }),
        };
        if let Ok(event) = serde_json::to_value(event) {
            let _ = store.write_event(None, event).await;
        }
    }
}

fn spawn_passthrough_response_recording(
    state: &PassthroughState,
    session_id: &Option<String>,
    request_id: &Option<String>,
    response: (u16, bytes::Bytes, axum::http::HeaderMap),
    timings: (std::time::Instant, std::time::Instant),
    user_agent: Option<String>,
) {
    let (Some(store), Some(session_id), Some(request_id)) = (
        state.session_store.clone(),
        session_id.clone(),
        request_id.clone(),
    ) else {
        return;
    };
    let (status, bytes, headers) = response;
    let total_duration_ms = timings.0.elapsed().as_millis() as u64;
    let provider_time_ms = timings.1.elapsed().as_millis() as u64;
    let timestamp = chrono::Utc::now();
    let encoding = headers
        .get_all("content-encoding")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect::<Vec<_>>()
        .join(",");
    let upstream_request_id = headers
        .get("request-id")
        .and_then(|value| value.to_str().ok())
        .map(|value| format!(", request_id={value}"))
        .unwrap_or_default();

    tokio::spawn(async move {
        use lunaroute_session::{
            SessionEvent,
            events::{FinalSessionStats, TokenTotals, ToolStats, ToolUsageSummary},
        };
        let mut tokens = TokenTotals::default();
        let mut tool_summary = ToolUsageSummary::default();
        let mut finish_reason = None;
        let mut error_type = None;
        let parse_result = std::panic::AssertUnwindSafe(async {
            let (response, response_size_bytes) = tokio::task::spawn_blocking(move || {
                // Recording has its own budget; compressed bytes are forwarded unchanged.
                let decoded = crate::async_stream_parser::decode_response_body(
                    &bytes,
                    &encoding,
                    32 * 1024 * 1024,
                )?;
                let response = serde_json::from_slice::<serde_json::Value>(&decoded)
                    .map_err(|error| format!("invalid response JSON: {error}"))?;
                Ok::<_, String>((response, decoded.len()))
            })
            .await
            .map_err(|error| format!("response parser task failed: {error}"))??;
            error_type = response
                .pointer("/error/type")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            finish_reason = response
                .get("stop_reason")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            if let Some(usage) = response.get("usage") {
                crate::async_stream_parser::update_anthropic_usage(&mut tokens, usage);
                tokens.total_thinking = usage
                    .get("thinking_tokens")
                    .or_else(|| usage.get("extended_thinking_tokens"))
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
            }
            tokens.grand_total = tokens.total_input
                + tokens.total_output
                + tokens.total_cache_read
                + tokens.total_cache_creation;
            let content = response.get("content").and_then(|value| value.as_array());
            for block in content.into_iter().flatten() {
                if block.get("type").and_then(|value| value.as_str()) != Some("tool_use") {
                    continue;
                }
                let Some(name) = block.get("name").and_then(|value| value.as_str()) else {
                    continue;
                };
                tool_summary
                    .by_tool
                    .entry(name.to_string())
                    .or_insert_with(|| ToolStats {
                        call_count: 0,
                        total_execution_time_ms: 0,
                        avg_execution_time_ms: 0,
                        error_count: 0,
                    })
                    .call_count += 1;
                tool_summary.total_tool_calls += 1;
                if let Some(id) = block.get("id").and_then(|value| value.as_str()) {
                    let arguments = block
                        .get("input")
                        .unwrap_or(&serde_json::Value::Null)
                        .to_string();
                    let event = SessionEvent::ToolCallRecorded {
                        session_id: session_id.clone(),
                        request_id: request_id.clone(),
                        timestamp,
                        tool_name: name.to_string(),
                        tool_call_id: id.to_string(),
                        execution_time_ms: None,
                        input_size_bytes: arguments.len(),
                        output_size_bytes: None,
                        success: None,
                        tool_arguments: Some(arguments),
                    };
                    if let Ok(event) = serde_json::to_value(event) {
                        let _ = store.write_event(None, event).await;
                    }
                }
            }
            tool_summary.unique_tool_count = tool_summary.by_tool.len() as u32;
            let event = SessionEvent::StatsUpdated {
                session_id: session_id.clone(),
                request_id: request_id.clone(),
                timestamp,
                token_updates: Some(tokens.clone()),
                tool_call_updates: Some(tool_summary.clone()),
                model_used: response
                    .get("model")
                    .and_then(|value| value.as_str())
                    .map(str::to_string),
                response_size_bytes,
                content_blocks: content.map_or(0, Vec::len),
                has_refusal: finish_reason.as_deref() == Some("refusal"),
                user_agent,
            };
            if let Ok(event) = serde_json::to_value(event) {
                let _ = store.write_event(None, event).await;
            }
            Ok::<_, String>(())
        });
        let parse_error = match futures::FutureExt::catch_unwind(parse_result).await {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(_) => Some("response parser panicked".to_string()),
        };
        let mut error = if !(200..300).contains(&status) {
            Some(format!(
                "Anthropic HTTP {status}{}{}",
                error_type
                    .map(|kind| format!(" ({kind})"))
                    .unwrap_or_default(),
                upstream_request_id
            ))
        } else {
            None
        };
        if let Some(parse_error) = parse_error {
            let message = format!(
                "Anthropic response recording failed (HTTP {status}{upstream_request_id}): {parse_error}"
            );
            tracing::warn!(session_id, request_id, "{message}");
            error = Some(match error {
                Some(error) => format!("{error}; {message}"),
                None => message,
            });
        }
        let event = SessionEvent::Completed {
            session_id,
            request_id,
            timestamp,
            success: error.is_none(),
            error,
            finish_reason,
            final_stats: Box::new(FinalSessionStats {
                total_duration_ms,
                provider_time_ms,
                proxy_overhead_ms: total_duration_ms.saturating_sub(provider_time_ms) as f64,
                total_tokens: tokens,
                tool_summary,
                performance: Default::default(),
                streaming_stats: None,
                estimated_cost: None,
            }),
        };
        if let Ok(event) = serde_json::to_value(event) {
            let _ = store.write_event(None, event).await;
        }
    });
}

/// Passthrough handler for Anthropic→Anthropic routing (no normalization)
/// Takes raw JSON, sends directly to Anthropic, returns raw JSON
/// Preserves 100% API fidelity while still extracting metrics
pub async fn messages_passthrough(
    State(state): State<Arc<PassthroughState>>,
    headers: axum::http::HeaderMap,
    RawQuery(query): RawQuery,
    Json(req): Json<serde_json::Value>,
) -> Result<Response, IngressError> {
    let start_time = std::time::Instant::now();
    tracing::debug!("Anthropic passthrough mode: skipping normalization");

    let mut req = req;

    // LUNAROUTE marker detection — check for provider override
    let marker_result = crate::marker::extract_marker(&req);
    let mut override_connector: Option<Arc<lunaroute_egress::anthropic::AnthropicConnector>> = None;
    let mut cross_dialect_connector: Option<Arc<lunaroute_egress::openai::OpenAIConnector>> = None;
    let mut marker_provider_name: Option<String> = None;

    match &marker_result {
        crate::marker::MarkerResult::Provider(name) => {
            if let Some(registry) = &state.provider_registry {
                if let Some(entry) = registry.get(name) {
                    if entry.connector_type != crate::ProviderType::Anthropic {
                        // Cross-dialect: Anthropic request → OpenAI provider
                        if let Some(ref connector) = entry.openai_connector {
                            tracing::info!(
                                "LUNAROUTE marker: cross-dialect routing to OpenAI provider '{}', model_override={:?}",
                                name,
                                entry.model_override
                            );
                            cross_dialect_connector = Some(connector.clone());
                            marker_provider_name = Some(name.clone());

                            // Apply model override
                            if let Some(ref model) = entry.model_override {
                                req["model"] = serde_json::Value::String(model.clone());
                            }
                        } else {
                            tracing::warn!(
                                "LUNAROUTE marker '{}' targets OpenAI provider but no OpenAI connector available",
                                name
                            );
                            return Err(IngressError::InvalidRequest(format!(
                                "LUNAROUTE marker targets provider '{}' but no OpenAI connector is configured for it",
                                name
                            )));
                        }
                    } else if let Some(ref connector) = entry.anthropic_connector {
                        tracing::info!(
                            "LUNAROUTE marker: routing to provider '{}', model_override={:?}",
                            name,
                            entry.model_override
                        );
                        override_connector = Some(connector.clone());
                        marker_provider_name = Some(name.clone());

                        // Apply model override
                        if let Some(ref model) = entry.model_override {
                            req["model"] = serde_json::Value::String(model.clone());
                        }
                    }
                } else {
                    tracing::warn!(
                        "LUNAROUTE marker references unknown provider '{}', using default",
                        name
                    );
                }
            } else {
                tracing::debug!(
                    "LUNAROUTE marker '{}' found but no provider registry available, using default",
                    name
                );
            }
            crate::marker::strip_marker(&mut req);
        }
        crate::marker::MarkerResult::Clear => {
            crate::marker::strip_marker(&mut req);
        }
        crate::marker::MarkerResult::None => {}
    }

    // Extract session ID from metadata.user_id. The shape has changed over time
    // (legacy "user_..._session_<uuid>" flat string, current JSON-encoded object);
    // extract_session_id_from_user_id handles all known shapes.
    let session_id = req
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(extract_session_id_from_user_id);

    let mut request_headers = forwardable_headers(&headers);
    request_headers.remove("host");
    let passthrough_headers = request_headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();

    let is_streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let before_provider = std::time::Instant::now();
    let pre_provider_overhead = before_provider.duration_since(start_time);
    tracing::debug!(
        "Proxy overhead before provider call: {:.2}ms",
        pre_provider_overhead.as_secs_f64() * 1000.0
    );

    // Extract model before req is moved (for metrics and session recording)
    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Extract user-agent from headers for session tracking
    // Truncate to 255 chars to prevent database issues with extremely long user agents
    let user_agent = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            if s.len() > 255 {
                s.chars().take(255).collect()
            } else {
                s.to_string()
            }
        });

    // Use extracted session ID from metadata, or generate a new one if not available
    let recording_session_id = if state.session_store.is_some() {
        session_id
            .clone()
            .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
    } else {
        None
    };

    let recording_request_id = if state.session_store.is_some() {
        Some(uuid::Uuid::new_v4().to_string())
    } else {
        None
    };

    // Start session recording if enabled (using async events)
    if let (Some(session_store), Some(session_id), Some(request_id)) = (
        &state.session_store,
        &recording_session_id,
        &recording_request_id,
    ) {
        use lunaroute_session::{SessionEvent, events::SessionMetadata as V2Metadata};

        // Enqueue Started before spawning dependent events. SessionStore writes
        // are non-blocking, and preserving this order prevents orphan rows.
        let session_tags_clone = {
            let mut tags = vec![];
            if let Some(ref provider_name) = marker_provider_name {
                tags.push(format!("lunaroute:{}", provider_name));
                if let Some(registry) = &state.provider_registry
                    && let Some(entry) = registry.get(provider_name)
                    && let Some(ref model) = entry.model_override
                {
                    tags.push(format!("model_override:{}", model));
                }
            }
            tags
        };
        let event = SessionEvent::Started {
            session_id: session_id.clone(),
            request_id: request_id.clone(),
            timestamp: chrono::Utc::now(),
            model_requested: model.clone(),
            provider: marker_provider_name
                .as_deref()
                .unwrap_or("anthropic")
                .to_string(),
            listener: "anthropic".to_string(),
            is_streaming,
            metadata: V2Metadata {
                client_ip: None,
                user_agent: user_agent.clone(),
                api_version: None,
                request_headers: Default::default(),
                session_tags: session_tags_clone,
            },
        };
        if let Ok(json) = serde_json::to_value(event) {
            let _ = session_store.write_event(None, json).await;
        }

        // Record the request
        use lunaroute_session::events::RequestStats;

        // Extract request text from messages (best effort for passthrough mode)
        let request_text = req
            .get("messages")
            .and_then(|m| m.as_array())
            .and_then(|msgs| msgs.last())
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let message_count = req
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let has_system_prompt = req.get("system").is_some();
        let has_tools = req.get("tools").is_some();
        let tool_count = req
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        // Record RequestRecorded event (spawn async to avoid blocking)
        let store_clone = session_store.clone();
        let session_id_clone = session_id.clone();
        let request_id_clone = request_id.clone();
        let req_size = serde_json::to_string(&req).map(|s| s.len()).unwrap_or(0);
        let req_clone = req.clone();

        // Clone again for the second spawn (tool result extraction)
        let store_clone_for_tools = store_clone.clone();
        let session_id_clone_for_tools = session_id_clone.clone();
        let request_id_clone_for_tools = request_id_clone.clone();

        tokio::spawn(async move {
            let event = SessionEvent::RequestRecorded {
                session_id: session_id_clone,
                request_id: request_id_clone,
                timestamp: chrono::Utc::now(),
                request_text,
                request_json: req_clone,
                estimated_tokens: 0, // Not available in passthrough mode
                stats: RequestStats {
                    pre_processing_ms: pre_provider_overhead.as_secs_f64() * 1000.0,
                    request_size_bytes: req_size,
                    message_count,
                    has_system_prompt,
                    has_tools,
                    tool_count,
                },
            };
            if let Ok(json) = serde_json::to_value(event) {
                let _ = store_clone.write_event(None, json).await;
            }
        });

        // Extract and record tool results from incoming messages
        // Use ToolCallMapper to look up tool names from tool_use_id
        let tool_call_mapper_clone = state.tool_call_mapper.clone();
        let req_for_tools = req.clone();
        tokio::spawn(async move {
            // Check for messages array
            if let Some(messages) = req_for_tools.get("messages").and_then(|m| m.as_array()) {
                for message in messages {
                    // Check for content blocks
                    if let Some(content_blocks) = message.get("content").and_then(|c| c.as_array())
                    {
                        for block in content_blocks {
                            // Look for ToolResult blocks
                            if let Some("tool_result") = block.get("type").and_then(|t| t.as_str())
                            {
                                let tool_use_id =
                                    block.get("tool_use_id").and_then(|id| id.as_str());
                                let content = block.get("content").and_then(|c| c.as_str());
                                let is_error = block.get("is_error").and_then(|e| e.as_bool());

                                if let Some(tool_use_id) = tool_use_id {
                                    // Look up tool name from mapper
                                    let tool_name = tool_call_mapper_clone
                                        .lookup(tool_use_id)
                                        .unwrap_or_else(|| "unknown".to_string());

                                    // Calculate output size
                                    let output_size = content.map(|c| c.len()).unwrap_or(0);

                                    // Map is_error to success (success = !is_error)
                                    let success = is_error.map(|err| !err);

                                    // Emit ToolCallRecorded event with result info
                                    let event = lunaroute_session::SessionEvent::ToolCallRecorded {
                                        session_id: session_id_clone_for_tools.clone(),
                                        request_id: request_id_clone_for_tools.clone(),
                                        timestamp: chrono::Utc::now(),
                                        tool_name,
                                        tool_call_id: tool_use_id.to_string(),
                                        execution_time_ms: None,
                                        input_size_bytes: 0, // We don't have input size here
                                        output_size_bytes: Some(output_size),
                                        success,
                                        tool_arguments: None, // Already recorded when tool was called
                                    };
                                    if let Ok(json) = serde_json::to_value(event) {
                                        let _ = store_clone_for_tools.write_event(None, json).await;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    // Cross-dialect routing: Anthropic request → OpenAI provider via normalization
    // TODO: Add response recording for cross-dialect path (request is already recorded above,
    // but response/tool-call recording is skipped because this returns early before the
    // passthrough recording flow). Needs ResponseRecorded event emission for non-streaming
    // and streaming tool-call tracking for streaming mode.
    if let Some(ref cd_connector) = cross_dialect_connector {
        let typed_req: AnthropicMessagesRequest =
            serde_json::from_value(req.clone()).map_err(|e| {
                IngressError::InvalidRequest(format!(
                    "Failed to parse request for cross-dialect routing: {}",
                    e
                ))
            })?;

        if is_streaming {
            let mut normalized = to_normalized(typed_req)?;
            normalized.stream = true;

            let event_stream = cd_connector
                .stream(normalized)
                .await
                .map_err(|e| IngressError::ProviderError(e.to_string()))?;

            let stream_id = Arc::new(format!("msg_{}", uuid::Uuid::new_v4().simple()));
            let model_arc = Arc::new(model.clone());

            // OpenAI's stream doesn't emit NormalizedStreamEvent::Start, but Anthropic
            // clients expect a message_start event. Prepend a synthetic Start event.
            let start_event =
                futures::stream::once(futures::future::ready(Ok(NormalizedStreamEvent::Start {
                    id: stream_id.as_str().to_string(),
                    model: model_arc.as_str().to_string(),
                })));
            let full_stream = start_event.chain(event_stream);

            let sse_stream = full_stream
                .scan(StreamBlockState::default(), move |block_state, result| {
                    let stream_id = Arc::clone(&stream_id);
                    let model = Arc::clone(&model_arc);
                    let sse_events: Vec<Result<Event, IngressError>> = match result {
                        Ok(event) => {
                            let events = stream_event_to_anthropic_events(
                                event,
                                stream_id.as_str(),
                                model.as_str(),
                                block_state,
                            );
                            events
                                .into_iter()
                                .map(|evt| match Event::default().json_data(evt) {
                                    Ok(event) => Ok::<_, IngressError>(event),
                                    Err(e) => Err(IngressError::Internal(format!(
                                        "Failed to create SSE event: {}",
                                        e
                                    ))),
                                })
                                .collect()
                        }
                        Err(e) => {
                            tracing::error!("Cross-dialect stream error: {}", e);
                            let mut error_events = Vec::new();

                            // Close any active content block before the error
                            if let Some(idx) = block_state.active_index.take()
                                && let Ok(evt) = Event::default().json_data(
                                    AnthropicStreamEvent::ContentBlockStop { index: idx },
                                )
                            {
                                error_events.push(Ok(evt));
                            }

                            // Send the error event
                            let error_payload = serde_json::json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": e.to_string()
                                }
                            });
                            match Event::default().json_data(error_payload) {
                                Ok(event) => error_events.push(Ok(event)),
                                Err(_) => error_events.push(Err(IngressError::Internal(
                                    "Failed to create error SSE event".to_string(),
                                ))),
                            }

                            // Send message_stop to properly terminate the stream
                            if let Ok(evt) =
                                Event::default().json_data(AnthropicStreamEvent::MessageStop)
                            {
                                error_events.push(Ok(evt));
                            }

                            error_events
                        }
                    };
                    futures::future::ready(Some(futures::stream::iter(sse_events)))
                })
                .flatten();

            let sse_keepalive = if state.sse_keepalive_enabled {
                KeepAlive::new().interval(std::time::Duration::from_secs(
                    state.sse_keepalive_interval_secs,
                ))
            } else {
                KeepAlive::new().interval(std::time::Duration::from_secs(86400))
            };

            return Ok(Sse::new(sse_stream)
                .keep_alive(sse_keepalive)
                .into_response());
        } else {
            let mut normalized = to_normalized(typed_req)?;
            normalized.stream = false;

            let normalized_resp = cd_connector
                .send(normalized)
                .await
                .map_err(|e| IngressError::ProviderError(e.to_string()))?;

            let anthropic_resp = from_normalized(normalized_resp);
            return Ok(Json(anthropic_resp).into_response());
        }
    }

    if is_streaming {
        // Handle streaming passthrough
        let connector = override_connector.as_ref().unwrap_or(&state.connector);
        let stream_response = match connector
            .stream_passthrough_with_query(req, passthrough_headers, query.as_deref())
            .await
        {
            Ok(response) => response,
            Err(error) => {
                record_passthrough_error(
                    &state,
                    &recording_session_id,
                    &recording_request_id,
                    start_time,
                    error.to_string(),
                )
                .await;
                return Err(IngressError::ProviderError(error.to_string()));
            }
        };

        // If the response is an error (non-2xx), pass it through as a raw response
        // instead of trying to set up SSE streaming. Error responses from Anthropic
        // are regular JSON, not SSE streams, and clients expect them in native format.
        if !stream_response.status().is_success() {
            let status_code = stream_response.status().as_u16();
            let response_headers = stream_response.headers().clone();
            let body_bytes = match stream_response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    let error = format!(
                        "Failed to read Anthropic HTTP {status_code} response body: {error}"
                    );
                    record_passthrough_error(
                        &state,
                        &recording_session_id,
                        &recording_request_id,
                        start_time,
                        error.clone(),
                    )
                    .await;
                    return Err(IngressError::ProviderError(error));
                }
            };
            spawn_passthrough_response_recording(
                &state,
                &recording_session_id,
                &recording_request_id,
                (status_code, body_bytes.clone(), response_headers.clone()),
                (start_time, before_provider),
                user_agent,
            );

            let mut axum_response = axum::http::Response::builder()
                .status(status_code)
                .body(axum::body::Body::from(body_bytes))
                .map_err(|e| {
                    IngressError::Internal(format!("Failed to build error response: {}", e))
                })?;
            *axum_response.headers_mut() = forwardable_headers(&response_headers);
            return Ok(axum_response);
        }

        let mut response_headers = forwardable_headers(stream_response.headers());
        // Axum reconstructs SSE, so only its content metadata describes the new body.
        response_headers.remove("content-encoding");
        response_headers.remove("content-type");

        // Track streaming metrics using shared module
        use crate::streaming_metrics::StreamingMetricsTracker;
        use futures::StreamExt;

        let tracker = StreamingMetricsTracker::new(before_provider);
        let tracker_ref = tracker.clone();
        let tracker_for_finalize = tracker.clone();
        let metrics_clone = state.metrics.clone();
        let model_clone = model.clone();

        let byte_stream = stream_response.bytes_stream();
        let sse_stream = eventsource_stream::EventStream::new(byte_stream);

        let stream_status = Arc::new(std::sync::Mutex::new(PassthroughStreamStatus::default()));
        let parser_sender = if let (Some(session_store), Some(session_id), Some(request_id)) = (
            state.session_store.clone(),
            recording_session_id.clone(),
            recording_request_id.clone(),
        ) {
            // Backpressure bounds queued events while retaining usage and tools from the whole stream.
            let (sender, receiver) = tokio::sync::mpsc::channel::<
                Result<
                    eventsource_stream::Event,
                    eventsource_stream::EventStreamError<std::convert::Infallible>,
                >,
            >(128);
            let events = futures::stream::unfold(receiver, |mut receiver| async move {
                receiver.recv().await.map(|event| (event, receiver))
            })
            .boxed();
            let parser_task = crate::async_stream_parser::spawn_anthropic_parser(
                events,
                session_id.clone(),
                request_id.clone(),
                session_store.clone(),
                user_agent.clone(),
                Some(state.tool_call_mapper.clone()),
            );
            let status = stream_status.clone();
            let metrics = state.metrics.clone();
            let model = model.clone();
            tokio::spawn(async move {
                use lunaroute_session::{
                    SessionEvent,
                    events::{
                        FinalSessionStats, PerformanceMetrics, TokenTotals, ToolUsageSummary,
                    },
                };

                let parser_result = parser_task.await;
                let error = {
                    let status = status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    status.error.clone().or_else(|| {
                        if let Err(error) = parser_result {
                            Some(format!("Anthropic stream recording failed: {error}"))
                        } else if !status.message_stopped {
                            Some("Anthropic stream ended before message_stop".to_string())
                        } else {
                            None
                        }
                    })
                };
                let finalized = tracker_for_finalize.finalize(start_time, before_provider);
                finalized.record_to_prometheus(&metrics, "anthropic", &model);

                if finalized.ttft_ms > 0 {
                    let event = SessionEvent::StreamStarted {
                        session_id: session_id.clone(),
                        request_id: request_id.clone(),
                        timestamp: chrono::Utc::now(),
                        time_to_first_token_ms: finalized.ttft_ms,
                    };
                    if let Ok(json) = serde_json::to_value(event) {
                        let _ = session_store.write_event(None, json).await;
                    }
                }
                let event = SessionEvent::Completed {
                    session_id,
                    request_id,
                    timestamp: chrono::Utc::now(),
                    success: error.is_none(),
                    error,
                    finish_reason: finalized.finish_reason.clone(),
                    final_stats: Box::new(FinalSessionStats {
                        total_duration_ms: finalized.total_duration_ms,
                        provider_time_ms: finalized.total_duration_ms,
                        proxy_overhead_ms: 0.0,
                        total_tokens: TokenTotals::default(),
                        tool_summary: ToolUsageSummary::default(),
                        performance: PerformanceMetrics::default(),
                        streaming_stats: Some(finalized.to_streaming_stats()),
                        estimated_cost: None,
                    }),
                };
                if let Ok(json) = serde_json::to_value(event) {
                    let _ = session_store.write_event(None, json).await;
                }
            });
            Some(sender)
        } else {
            None
        };

        let tracked_stream = sse_stream.then(move |event_result| {
            let parser_sender = parser_sender.clone();
            let tracker_ref = tracker_ref.clone();
            let metrics_clone = metrics_clone.clone();
            let model_clone = model_clone.clone();
            let stream_status = stream_status.clone();
            async move {
                match event_result {
                    Ok(event) => {
                        let now = std::time::Instant::now();

                        // Track TTFT on first chunk
                        tracker_ref.record_ttft(now);

                        // Track chunk latency with memory bounds
                        let _ = tracker_ref.record_chunk_latency(
                            now,
                            "anthropic",
                            &model_clone,
                            &metrics_clone,
                        );

                        // Increment chunk count
                        tracker_ref.increment_chunk_count();

                        // Parse event data to extract text and metadata
                        if let Ok(anthropic_event) =
                            serde_json::from_str::<serde_json::Value>(&event.data)
                        {
                            {
                                let mut status = stream_status
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                                match anthropic_event.get("type").and_then(|value| value.as_str()) {
                                    Some("error") => {
                                        status.error.get_or_insert_with(|| {
                                            anthropic_event
                                                .get("error")
                                                .unwrap_or(&anthropic_event)
                                                .to_string()
                                        });
                                    }
                                    Some("message_stop") => status.message_stopped = true,
                                    _ => {}
                                }
                            }
                            // Extract text deltas (with size limit to prevent OOM)
                            if let Some("content_block_delta") =
                                anthropic_event.get("type").and_then(|t| t.as_str())
                                && let Some("text_delta") = anthropic_event
                                    .get("delta")
                                    .and_then(|d| d.get("type"))
                                    .and_then(|t| t.as_str())
                                && let Some(text) = anthropic_event
                                    .get("delta")
                                    .and_then(|d| d.get("text"))
                                    .and_then(|t| t.as_str())
                            {
                                tracker_ref.accumulate_text(
                                    text,
                                    "anthropic",
                                    &model_clone,
                                    &metrics_clone,
                                );
                            }

                            // Extract model from message_start
                            if let Some("message_start") =
                                anthropic_event.get("type").and_then(|t| t.as_str())
                                && let Some(model_str) = anthropic_event
                                    .get("message")
                                    .and_then(|m| m.get("model"))
                                    .and_then(|m| m.as_str())
                            {
                                tracker_ref.set_model(model_str.to_string());
                            }

                            // Extract finish reason from message_delta
                            if let Some("message_delta") =
                                anthropic_event.get("type").and_then(|t| t.as_str())
                                && let Some(reason) = anthropic_event
                                    .get("delta")
                                    .and_then(|d| d.get("stop_reason"))
                                    .and_then(|r| r.as_str())
                            {
                                tracker_ref.set_finish_reason(reason.to_string());
                            }
                        }

                        if let Some(sender) = parser_sender {
                            let _ = sender.send(Ok(event.clone())).await;
                        }

                        // Forward the event exactly as received (preserve event type, id, data)
                        let mut sse_event = Event::default();

                        // Preserve event type (e.g., "message_start", "content_block_delta")
                        if !event.event.is_empty() {
                            sse_event = sse_event.event(event.event.clone());
                        }

                        // Preserve event ID if present
                        if !event.id.is_empty() {
                            sse_event = sse_event.id(event.id.clone());
                        }

                        // Forward data as-is (don't re-parse/re-serialize)
                        Ok(sse_event.data(event.data.clone()))
                    }
                    Err(e) => {
                        let error = format!("SSE stream error: {e}");
                        stream_status
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .error
                            .get_or_insert(error.clone());
                        Err(IngressError::Internal(error))
                    }
                }
            }
        });

        // Create SSE response with configured keepalive
        let sse_keepalive = if state.sse_keepalive_enabled {
            KeepAlive::new().interval(std::time::Duration::from_secs(
                state.sse_keepalive_interval_secs,
            ))
        } else {
            // Disable keepalive by setting a very long interval
            KeepAlive::new().interval(std::time::Duration::from_secs(86400)) // 24 hours
        };

        let sse_response = Sse::new(tracked_stream).keep_alive(sse_keepalive);
        let mut response = sse_response.into_response();

        // Add the response headers from Anthropic
        response.headers_mut().extend(response_headers);

        return Ok(response);
    }

    // Send directly to Anthropic API (non-streaming)
    let connector = override_connector.as_ref().unwrap_or(&state.connector);
    let response_result = connector
        .send_passthrough_with_query(req, passthrough_headers, query.as_deref())
        .await;

    let (response_status, raw_response_bytes, response_headers) = match response_result {
        Ok((status, bytes, headers)) => (status, bytes, headers),
        Err(error) => {
            record_passthrough_error(
                &state,
                &recording_session_id,
                &recording_request_id,
                start_time,
                error.to_string(),
            )
            .await;
            return Err(IngressError::ProviderError(error.to_string()));
        }
    };

    let after_provider = std::time::Instant::now();
    let provider_time = after_provider.duration_since(before_provider);
    tracing::debug!(
        "Provider response time: {:.2}ms",
        provider_time.as_secs_f64() * 1000.0
    );

    // Preserve the upstream status and encoded bytes; Axum frames the new HTTP response.
    let mut axum_response = axum::http::Response::builder()
        .status(response_status)
        .body(axum::body::Body::from(raw_response_bytes.clone()))
        .map_err(|e| IngressError::Internal(format!("Failed to build response: {}", e)))?;

    *axum_response.headers_mut() = forwardable_headers(&response_headers);
    spawn_passthrough_response_recording(
        &state,
        &recording_session_id,
        &recording_request_id,
        (response_status, raw_response_bytes, response_headers),
        (start_time, before_provider),
        user_agent,
    );

    Ok(axum_response)
}

/// Create Anthropic router with provider state
pub fn router(provider: Arc<dyn Provider>) -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
        .with_state(provider)
}

/// Create Anthropic router with session recording for normalized routed requests.
pub fn router_with_session_store(
    provider: Arc<dyn Provider>,
    session_store: Arc<dyn SessionStore>,
    provider_name: impl Into<String>,
    listener_name: impl Into<String>,
) -> Router {
    let recording_provider = Arc::new(lunaroute_session::SessionStoreRecordingProvider::new(
        provider,
        session_store,
        provider_name,
        listener_name,
    ));
    router(recording_provider)
}

/// Create Anthropic passthrough router (for Anthropic→Anthropic direct routing)
#[allow(clippy::too_many_arguments)]
pub fn passthrough_router(
    connector: Arc<lunaroute_egress::anthropic::AnthropicConnector>,
    stats_tracker: Option<Arc<dyn crate::types::SessionStatsTracker>>,
    metrics: Option<Arc<lunaroute_observability::Metrics>>,
    session_store: Option<Arc<dyn SessionStore>>,
    sse_keepalive_interval_secs: u64,
    sse_keepalive_enabled: bool,
    provider_registry: Option<Arc<crate::ProviderRegistry>>,
) -> Router {
    let state = Arc::new(PassthroughState {
        connector,
        stats_tracker,
        metrics,
        session_store,
        tool_call_mapper: Arc::new(lunaroute_session::ToolCallMapper::new()),
        sse_keepalive_interval_secs,
        sse_keepalive_enabled,
        provider_registry,
    });

    Router::new()
        .route("/v1/messages", post(messages_passthrough))
        .layer(tower_http::compression::CompressionLayer::new())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::stream;
    use lunaroute_core::provider::ProviderCapabilities;

    // Mock provider for testing
    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn send(
            &self,
            _request: NormalizedRequest,
        ) -> lunaroute_core::Result<NormalizedResponse> {
            Ok(NormalizedResponse {
                id: "test-123".to_string(),
                model: "claude-3-opus".to_string(),
                choices: vec![Choice {
                    index: 0,
                    message: Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("Hello from mock".to_string()),
                        name: None,
                        tool_calls: vec![],
                        tool_call_id: None,
                    },
                    finish_reason: Some(FinishReason::Stop),
                }],
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                created: 1234567890,
                metadata: std::collections::HashMap::new(),
            })
        }

        async fn stream(
            &self,
            _request: NormalizedRequest,
        ) -> lunaroute_core::Result<
            Box<
                dyn futures::Stream<
                        Item = lunaroute_core::Result<
                            lunaroute_core::normalized::NormalizedStreamEvent,
                        >,
                    > + Send
                    + Unpin,
            >,
        > {
            Ok(Box::new(stream::empty()))
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_streaming: false,
                supports_tools: false,
                supports_vision: false,
            }
        }
    }

    #[test]
    fn test_to_normalized() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: Some(AnthropicMessageContent::Text("Hello!".to_string())),
            }],
            system: Some(AnthropicSystem::Text(
                "You are a helpful assistant".to_string(),
            )),
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stream: Some(false),
            stop_sequences: None,
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        assert_eq!(normalized.model, "claude-3-opus");
        assert_eq!(normalized.messages.len(), 1);
        assert_eq!(
            normalized.system,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(normalized.max_tokens, Some(1024));
    }

    #[test]
    fn test_from_normalized() {
        let resp = NormalizedResponse {
            id: "test-123".to_string(),
            model: "claude-3-opus".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text("Hello, world!".to_string()),
                    name: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            created: 1234567890,
            metadata: std::collections::HashMap::new(),
        };

        let anthropic = from_normalized(resp);
        assert_eq!(anthropic.model, "claude-3-opus");
        assert_eq!(anthropic.role, "assistant");
        assert_eq!(anthropic.stop_reason, Some("end_turn".to_string()));
        assert_eq!(anthropic.usage.input_tokens, 10);
        assert_eq!(anthropic.usage.output_tokens, 5);
    }

    #[test]
    fn test_invalid_role() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![AnthropicMessage {
                role: "invalid".to_string(),
                content: Some(AnthropicMessageContent::Text("test".to_string())),
            }],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
        };

        let result = to_normalized(req);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_messages_endpoint() {
        let provider = Arc::new(MockProvider);
        let req = AnthropicMessagesRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: Some(AnthropicMessageContent::Text("Hello!".to_string())),
            }],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
        };

        let response = messages(State(provider), Json(req)).await;
        assert!(response.is_ok());
    }

    #[test]
    fn test_all_anthropic_finish_reasons() {
        // Test all finish reason mappings for Anthropic
        let finish_reasons = vec![
            (FinishReason::Stop, "end_turn"),
            (FinishReason::Length, "max_tokens"),
            (FinishReason::ToolCalls, "tool_use"),
            // ContentFilter and Error both map to end_turn for Anthropic
            (FinishReason::ContentFilter, "end_turn"),
            (FinishReason::Error, "end_turn"),
        ];

        for (reason, expected) in finish_reasons {
            let resp = NormalizedResponse {
                id: "test".to_string(),
                model: "claude-3-opus".to_string(),
                choices: vec![Choice {
                    index: 0,
                    message: Message {
                        role: Role::Assistant,
                        content: MessageContent::Text("test".to_string()),
                        name: None,
                        tool_calls: vec![],
                        tool_call_id: None,
                    },
                    finish_reason: Some(reason),
                }],
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                created: 1234567890,
                metadata: std::collections::HashMap::new(),
            };

            let anthropic = from_normalized(resp);
            assert_eq!(anthropic.stop_reason, Some(expected.to_string()));
        }
    }

    #[test]
    fn test_anthropic_empty_messages() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![],
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
        };

        // Validation should reject empty messages array
        let result = to_normalized(req);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("messages array cannot be empty")
        );
    }

    #[test]
    fn test_anthropic_with_system_prompt() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: Some(AnthropicMessageContent::Text("Hello!".to_string())),
            }],
            system: Some(AnthropicSystem::Text(
                "You are a helpful assistant".to_string(),
            )),
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            stream: Some(true),
            stop_sequences: Some(vec!["STOP".to_string()]),
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        assert_eq!(
            normalized.system,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(normalized.max_tokens, Some(1024));
        assert_eq!(normalized.temperature, Some(0.7));
        assert_eq!(normalized.top_p, Some(0.9));
        assert_eq!(normalized.top_k, Some(40));
        assert!(normalized.stream);
        assert_eq!(normalized.stop_sequences, vec!["STOP".to_string()]);
    }

    #[test]
    fn test_anthropic_stop_sequence_field() {
        let resp = NormalizedResponse {
            id: "test".to_string(),
            model: "claude-3-opus".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text("test".to_string()),
                    name: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            created: 1234567890,
            metadata: std::collections::HashMap::new(),
        };

        let anthropic = from_normalized(resp);
        // Currently returns None - will be populated when tracking is implemented
        assert_eq!(anthropic.stop_sequence, None);
    }

    #[test]
    fn test_anthropic_multimodal_parts() {
        use lunaroute_core::normalized::ContentPart;

        let resp = NormalizedResponse {
            id: "test".to_string(),
            model: "claude-3-opus".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Parts(vec![ContentPart::Text {
                        text: "I see an image".to_string(),
                    }]),
                    name: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            created: 1234567890,
            metadata: std::collections::HashMap::new(),
        };

        let anthropic = from_normalized(resp);
        // Now properly extracts text from multimodal Parts
        assert_eq!(anthropic.content.len(), 1);
        if let AnthropicContent::Text { text } = &anthropic.content[0] {
            assert_eq!(text, "I see an image");
        } else {
            panic!("Expected Text content block");
        }
    }

    #[test]
    fn test_anthropic_router_creation() {
        let provider = Arc::new(MockProvider);
        let router = router(provider);
        // Just verify it creates without panicking
        // The router is properly configured with /v1/messages endpoint
        drop(router);
    }

    #[test]
    fn test_anthropic_system_array_format() {
        // Test the array format for system parameter (used by Claude Code)
        let req = AnthropicMessagesRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: Some(AnthropicMessageContent::Text("Hello!".to_string())),
            }],
            system: Some(AnthropicSystem::Blocks(vec![
                AnthropicSystemBlock::Text {
                    text: "You are a helpful assistant.".to_string(),
                },
                AnthropicSystemBlock::Text {
                    text: "You are concise.".to_string(),
                },
            ])),
            max_tokens: Some(1024),
            temperature: None,
            top_p: None,
            top_k: None,
            stream: Some(false),
            stop_sequences: None,
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        // Multiple blocks should be joined with newlines
        assert_eq!(
            normalized.system,
            Some("You are a helpful assistant.\nYou are concise.".to_string())
        );
    }

    #[test]
    fn test_anthropic_system_deserialize_string() {
        // Test that string format still works
        let json = r#"{"model":"claude-3-opus","messages":[],"system":"You are helpful"}"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(json).unwrap();
        match req.system {
            Some(AnthropicSystem::Text(text)) => assert_eq!(text, "You are helpful"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_anthropic_system_deserialize_array() {
        // Test that array format works (Claude Code format)
        let json = r#"{"model":"claude-3-opus","messages":[],"system":[{"type":"text","text":"You are helpful"}]}"#;
        let req: AnthropicMessagesRequest = serde_json::from_str(json).unwrap();
        match req.system {
            Some(AnthropicSystem::Blocks(blocks)) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    AnthropicSystemBlock::Text { text } => assert_eq!(text, "You are helpful"),
                    AnthropicSystemBlock::Unknown(_) => panic!("Expected Text variant"),
                }
            }
            _ => panic!("Expected Blocks variant"),
        }
    }

    // Helper function for testing session ID extraction
    fn extract_and_validate_session_id(user_id: &str) -> Option<String> {
        super::extract_session_id_from_user_id(&serde_json::Value::String(user_id.to_string()))
    }

    #[test]
    fn test_session_id_extraction_from_json_object_user_id() {
        // Current Claude client format: user_id is a JSON-encoded object
        let user_id = serde_json::json!(
            r#"{"device_id":"6d7768e28857b8461693cf8c58547d1ce156a657b3ab25880fa0409a28a23fea","account_uuid":"92005441-c12f-433c-89b2-43bc95d78f69","session_id":"bdfe0e7b-af96-4629-95ae-22bf20dbd9ff"}"#
        );
        assert_eq!(
            super::extract_session_id_from_user_id(&user_id),
            Some("bdfe0e7b-af96-4629-95ae-22bf20dbd9ff".to_string())
        );
    }

    #[test]
    fn test_session_id_extraction_from_unstringified_object() {
        // Defensive: if a client sends metadata.user_id as a real JSON object
        let user_id = serde_json::json!({
            "device_id": "abc",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
        });
        assert_eq!(
            super::extract_session_id_from_user_id(&user_id),
            Some("550e8400-e29b-41d4-a716-446655440000".to_string())
        );
    }

    #[test]
    fn test_session_id_extraction_json_missing_field() {
        // JSON-shaped but no session_id → None, no fallback to substring parse
        let user_id = serde_json::json!(r#"{"device_id":"abc"}"#);
        assert_eq!(super::extract_session_id_from_user_id(&user_id), None);
    }

    #[test]
    fn test_session_id_extraction_json_with_bad_chars_rejected() {
        // session_id present but contains forbidden chars
        let user_id = serde_json::json!(r#"{"session_id":"../../../etc/passwd"}"#);
        assert_eq!(super::extract_session_id_from_user_id(&user_id), None);
    }

    #[test]
    fn test_session_id_extraction_valid() {
        // Valid session ID with marker
        let user_id = "user_abc123_account_def456_session_550e8400-e29b-41d4-a716-446655440000";
        let session_id = extract_and_validate_session_id(user_id);
        assert_eq!(
            session_id,
            Some("550e8400-e29b-41d4-a716-446655440000".to_string())
        );
    }

    #[test]
    fn test_session_id_extraction_no_marker() {
        // No session marker - should use entire user_id
        let user_id = "user_abc123_account_def456";
        let session_id = extract_and_validate_session_id(user_id);
        assert_eq!(session_id, Some(user_id.to_string()));
    }

    #[test]
    fn test_session_id_extraction_empty() {
        // Empty session ID after marker
        let user_id = "user_abc_session_";
        let session_id = extract_and_validate_session_id(user_id);
        assert_eq!(session_id, None);
    }

    #[test]
    fn test_session_id_validation_path_traversal() {
        // Path traversal attempts should be rejected
        let user_id = "user_abc_session_../../../etc/passwd";
        let session_id = extract_and_validate_session_id(user_id);
        assert_eq!(session_id, None);
    }

    #[test]
    fn test_session_id_validation_special_chars() {
        // Special characters should be rejected
        let test_cases = vec![
            "user_session_test;DROP_TABLE",
            "user_session_test/path",
            "user_session_test\\path",
            "user_session_test@host",
            "user_session_test$var",
        ];

        for user_id in test_cases {
            let session_id = extract_and_validate_session_id(user_id);
            assert_eq!(session_id, None, "Should reject: {}", user_id);
        }
    }

    #[test]
    fn test_session_id_validation_too_long() {
        // Session ID too long should be rejected
        let long_id = format!("user_session_{}", "a".repeat(256));
        let session_id = extract_and_validate_session_id(&long_id);
        assert_eq!(session_id, None);
    }

    #[test]
    fn test_session_id_validation_multiple_markers() {
        // Should use the last occurrence of _session_
        let user_id = "user_session_old_session_new_session_final-uuid-123";
        let session_id = extract_and_validate_session_id(user_id);
        assert_eq!(session_id, Some("final-uuid-123".to_string()));
    }

    // Tool failure detection tests
    #[test]
    fn test_tool_result_extraction_with_success() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet-20241022".to_string(),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Text("Use the tool".to_string())),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolUse {
                            id: "toolu_123".to_string(),
                            name: "get_data".to_string(),
                            input: serde_json::json!({"key": "value"}),
                        },
                    ])),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolResult {
                            tool_use_id: "toolu_123".to_string(),
                            content: "Data retrieved successfully".to_string(),
                            is_error: None, // No error
                        },
                    ])),
                },
            ],
            max_tokens: Some(1024),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        assert_eq!(normalized.tool_results.len(), 1);
        assert_eq!(normalized.tool_results[0].tool_call_id, "toolu_123");
        assert_eq!(
            normalized.tool_results[0].content,
            "Data retrieved successfully"
        );
        assert!(!normalized.tool_results[0].is_error); // Should not be an error
    }

    #[test]
    fn test_tool_result_extraction_with_explicit_error() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet-20241022".to_string(),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Text("Use the tool".to_string())),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolUse {
                            id: "toolu_456".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path": "/tmp/test.txt"}),
                        },
                    ])),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolResult {
                            tool_use_id: "toolu_456".to_string(),
                            content: "File not found: /tmp/test.txt".to_string(),
                            is_error: Some(true), // Explicit error flag
                        },
                    ])),
                },
            ],
            max_tokens: Some(1024),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        assert_eq!(normalized.tool_results.len(), 1);
        assert_eq!(normalized.tool_results[0].tool_call_id, "toolu_456");
        assert!(normalized.tool_results[0].is_error); // Should be detected as error
        assert!(
            normalized.tool_results[0]
                .content
                .contains("File not found")
        );
    }

    #[test]
    fn test_tool_result_extraction_multiple_results() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet-20241022".to_string(),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Text(
                        "Use multiple tools".to_string(),
                    )),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolUse {
                            id: "toolu_1".to_string(),
                            name: "tool_a".to_string(),
                            input: serde_json::json!({}),
                        },
                        AnthropicContentBlock::ToolUse {
                            id: "toolu_2".to_string(),
                            name: "tool_b".to_string(),
                            input: serde_json::json!({}),
                        },
                    ])),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolResult {
                            tool_use_id: "toolu_1".to_string(),
                            content: "Success".to_string(),
                            is_error: Some(false),
                        },
                        AnthropicContentBlock::ToolResult {
                            tool_use_id: "toolu_2".to_string(),
                            content: "Failed to execute".to_string(),
                            is_error: Some(true),
                        },
                    ])),
                },
            ],
            max_tokens: Some(1024),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        assert_eq!(normalized.tool_results.len(), 2);

        // First result should not be an error
        let result_1 = normalized
            .tool_results
            .iter()
            .find(|r| r.tool_call_id == "toolu_1")
            .unwrap();
        assert!(!result_1.is_error);

        // Second result should be an error
        let result_2 = normalized
            .tool_results
            .iter()
            .find(|r| r.tool_call_id == "toolu_2")
            .unwrap();
        assert!(result_2.is_error);
    }

    #[test]
    fn test_tool_result_extraction_mixed_content() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet-20241022".to_string(),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Text("Test".to_string())),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::ToolUse {
                            id: "toolu_789".to_string(),
                            name: "calculate".to_string(),
                            input: serde_json::json!({"x": 10, "y": 20}),
                        },
                    ])),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: Some(AnthropicMessageContent::Blocks(vec![
                        AnthropicContentBlock::Text {
                            text: "Here is some text".to_string(),
                        },
                        AnthropicContentBlock::ToolResult {
                            tool_use_id: "toolu_789".to_string(),
                            content: "30".to_string(),
                            is_error: None,
                        },
                        AnthropicContentBlock::Text {
                            text: "More text".to_string(),
                        },
                    ])),
                },
            ],
            max_tokens: Some(1024),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
        };

        let normalized = to_normalized(req).unwrap();
        // Should extract tool result even when mixed with text blocks
        assert_eq!(normalized.tool_results.len(), 1);
        assert_eq!(normalized.tool_results[0].tool_call_id, "toolu_789");
        assert_eq!(normalized.tool_results[0].content, "30");
        assert!(!normalized.tool_results[0].is_error);
    }

    #[test]
    fn test_cross_dialect_anthropic_to_normalized_roundtrip() {
        let raw_json = serde_json::json!({
            "model": "@cf/moonshotai/kimi-k2.5",
            "messages": [
                {"role": "user", "content": "Hello from Claude Code"}
            ],
            "max_tokens": 1024,
            "stream": false
        });

        let typed_req: AnthropicMessagesRequest = serde_json::from_value(raw_json).unwrap();
        assert_eq!(typed_req.model, "@cf/moonshotai/kimi-k2.5");

        let mut normalized = to_normalized(typed_req).unwrap();
        normalized.stream = false;
        assert_eq!(normalized.model, "@cf/moonshotai/kimi-k2.5");
        assert_eq!(normalized.messages.len(), 1);
        assert!(!normalized.stream);

        let normalized_resp = NormalizedResponse {
            id: "chatcmpl-test".to_string(),
            model: "@cf/moonshotai/kimi-k2.5".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text("Hello from Kimi".to_string()),
                    name: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
            created: 1234567890,
            metadata: std::collections::HashMap::new(),
        };

        let anthropic_resp = from_normalized(normalized_resp);
        assert_eq!(anthropic_resp.model, "@cf/moonshotai/kimi-k2.5");
        assert_eq!(anthropic_resp.role, "assistant");
        assert_eq!(anthropic_resp.stop_reason, Some("end_turn".to_string()));
        assert!(!anthropic_resp.content.is_empty());
        match &anthropic_resp.content[0] {
            AnthropicContent::Text { text } => assert_eq!(text, "Hello from Kimi"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_cross_dialect_stream_event_conversion() {
        use lunaroute_core::normalized::Delta;

        let mut block_state = StreamBlockState::default();

        let event = NormalizedStreamEvent::Delta {
            index: 0,
            delta: Delta {
                role: None,
                content: Some("Hello from Kimi".to_string()),
            },
        };

        let anthropic_events = stream_event_to_anthropic_events(
            event,
            "msg_test123",
            "@cf/moonshotai/kimi-k2.5",
            &mut block_state,
        );

        assert_eq!(block_state.active_index, Some(0));
        assert!(!block_state.active_is_tool);
        assert!(anthropic_events.len() >= 2);
    }

    #[test]
    fn test_cross_dialect_tool_use_roundtrip() {
        let raw_json = serde_json::json!({
            "model": "@cf/moonshotai/kimi-k2.5",
            "messages": [
                {"role": "user", "content": "What's the weather?"}
            ],
            "max_tokens": 1024,
            "stream": false,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }]
        });

        let typed_req: AnthropicMessagesRequest = serde_json::from_value(raw_json).unwrap();
        let normalized = to_normalized(typed_req).unwrap();

        assert_eq!(normalized.tools.len(), 1);
        assert_eq!(normalized.tools[0].function.name, "get_weather");

        let normalized_resp = NormalizedResponse {
            id: "chatcmpl-test".to_string(),
            model: "@cf/moonshotai/kimi-k2.5".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(String::new()),
                    name: None,
                    tool_calls: vec![ToolCall {
                        id: "call_abc123".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"San Francisco"}"#.to_string(),
                        },
                    }],
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage {
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
            },
            created: 1234567890,
            metadata: std::collections::HashMap::new(),
        };

        let anthropic_resp = from_normalized(normalized_resp);
        assert_eq!(anthropic_resp.stop_reason, Some("tool_use".to_string()));

        let tool_use = anthropic_resp
            .content
            .iter()
            .find(|c| matches!(c, AnthropicContent::ToolUse { .. }));
        assert!(tool_use.is_some(), "Expected tool_use block in response");

        if let Some(AnthropicContent::ToolUse { id, name, input }) = tool_use {
            assert_eq!(id, "toolu_call_abc123");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "San Francisco");
        }
    }

    #[test]
    fn test_cross_dialect_stream_error_produces_sse_error_event() {
        use lunaroute_core::normalized::Delta;

        let mut block_state = StreamBlockState::default();

        // First send a valid delta to start content block
        let event = NormalizedStreamEvent::Delta {
            index: 0,
            delta: Delta {
                role: None,
                content: Some("partial".to_string()),
            },
        };
        let events =
            stream_event_to_anthropic_events(event, "msg_test", "test-model", &mut block_state);
        assert_eq!(block_state.active_index, Some(0));
        assert!(!block_state.active_is_tool);
        assert!(!events.is_empty());

        // Verify error SSE event can be constructed (matching the cross-dialect error path)
        let error_event = serde_json::json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "upstream provider error"
            }
        });
        let sse_result = Event::default().json_data(error_event);
        assert!(
            sse_result.is_ok(),
            "Error SSE event should serialize successfully"
        );
    }

    #[test]
    fn test_cross_dialect_model_override_applied_before_extraction() {
        // Verify that model override applied to raw JSON is visible when extracting model
        let mut req = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1024
        });

        // Simulate model override (as done in marker match block)
        req["model"] = serde_json::Value::String("@cf/moonshotai/kimi-k2.5".to_string());

        // Simulate model extraction (as done later in handler)
        let model = req
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        assert_eq!(model, "@cf/moonshotai/kimi-k2.5");

        // Also verify normalization sees the overridden model
        let typed_req: AnthropicMessagesRequest = serde_json::from_value(req).unwrap();
        let normalized = to_normalized(typed_req).unwrap();
        assert_eq!(normalized.model, "@cf/moonshotai/kimi-k2.5");
    }

    #[test]
    fn test_unknown_content_blocks_are_skipped_during_normalization() {
        // Claude Code sends thinking/redacted_thinking blocks that we don't normalize
        let raw_json = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think about this..."},
                        {"type": "redacted_thinking", "data": "abc123"},
                        {"type": "text", "text": "Hi there!"}
                    ]
                },
                {
                    "role": "user",
                    "content": "Follow up question"
                }
            ],
            "max_tokens": 1024,
            "stream": false
        });

        let typed_req: AnthropicMessagesRequest = serde_json::from_value(raw_json).unwrap();
        assert_eq!(typed_req.messages.len(), 3);

        let normalized = to_normalized(typed_req).unwrap();
        assert_eq!(normalized.messages.len(), 3);
        // Assistant message should only contain the text block, not thinking blocks
        assert_eq!(
            normalized.messages[1].content,
            MessageContent::Text("Hi there!".to_string())
        );
    }

    #[test]
    fn test_tool_result_with_array_content() {
        // Anthropic API allows tool_result content to be an array of blocks
        let raw_json = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [
                                {"type": "text", "text": "Result line 1"},
                                {"type": "text", "text": "Result line 2"}
                            ]
                        }
                    ]
                }
            ],
            "max_tokens": 1024
        });

        let typed_req: AnthropicMessagesRequest = serde_json::from_value(raw_json).unwrap();
        let normalized = to_normalized(typed_req).unwrap();
        // Array content should be joined with newlines
        match &normalized.messages[0].content {
            MessageContent::Text(text) => {
                assert!(text.contains("Result line 1"));
                assert!(text.contains("Result line 2"));
            }
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_to_anthropic_tool_id() {
        // Already valid Anthropic ID — kept as-is (no hash suffix)
        assert_eq!(to_anthropic_tool_id("toolu_abc123"), "toolu_abc123");

        // OpenAI call_ format — no invalid chars, just prefix (no hash suffix)
        assert_eq!(to_anthropic_tool_id("call_abc123"), "toolu_call_abc123");

        // ID with invalid characters — replaced with _, hash suffix appended
        let id = to_anthropic_tool_id("call.abc:123");
        assert!(id.starts_with("toolu_call_abc_123_"));
        assert!(id.len() > "toolu_call_abc_123_".len()); // has hash suffix

        // Empty ID — generates UUID-based ID
        let id = to_anthropic_tool_id("");
        assert!(id.starts_with("toolu_"));
        assert!(id.len() > 6);

        // Hyphenated ID — hyphens are valid, no sanitization needed (no hash suffix)
        assert_eq!(
            to_anthropic_tool_id("chatcmpl-abc123"),
            "toolu_chatcmpl-abc123"
        );

        // toolu_ prefix with invalid chars — sanitize with hash, don't double-prefix
        let id = to_anthropic_tool_id("toolu_functions.Bash:0");
        assert!(id.starts_with("toolu_functions_Bash_0_"));

        // Already has toolu_ prefix from upstream, all valid chars — no hash suffix
        assert_eq!(
            to_anthropic_tool_id("toolu_functionsRead0"),
            "toolu_functionsRead0"
        );

        // Collision safety: different upstream IDs that sanitize to same base get
        // different hash suffixes
        let id_a = to_anthropic_tool_id("call.a:1");
        let id_b = to_anthropic_tool_id("call_a_1");
        // id_b has no invalid chars, so no hash suffix; id_a has invalid chars, gets hash
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn test_streaming_tool_call_block_lifecycle() {
        use lunaroute_core::normalized::{FinishReason, FunctionCallDelta};

        let mut block_state = StreamBlockState::default();

        // 1. Start event resets state
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::Start {
                id: "msg_test".to_string(),
                model: "test-model".to_string(),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert_eq!(block_state.active_index, None);
        assert_eq!(events.len(), 1); // message_start

        // 2. Tool call start — should open a block
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::ToolCallDelta {
                index: 0,
                tool_call_index: 0,
                id: Some("call_abc".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some("Bash".to_string()),
                    arguments: None,
                }),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert!(block_state.active_is_tool);
        assert!(block_state.active_index.is_some());
        // Should have ContentBlockStart (no prior block to close)
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStart { index: 0, .. }))
        );

        // 3. Tool call partial JSON
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::ToolCallDelta {
                index: 0,
                tool_call_index: 0,
                id: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some("{\"cmd\":\"ls\"}".to_string()),
                }),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert_eq!(events.len(), 1); // Just the delta
        assert!(matches!(
            &events[0],
            AnthropicStreamEvent::ContentBlockDelta { index: 0, .. }
        ));

        // 4. End event — should close the tool block at index 0
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::End {
                finish_reason: FinishReason::ToolCalls,
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert_eq!(block_state.active_index, None);
        // Should have ContentBlockStop at index 0, MessageDelta, MessageStop
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStop { index: 0 }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::MessageStop))
        );
    }

    #[test]
    fn test_streaming_text_to_tool_transition() {
        use lunaroute_core::normalized::{Delta, FinishReason, FunctionCallDelta};

        let mut block_state = StreamBlockState::default();

        // Start
        let _ = stream_event_to_anthropic_events(
            NormalizedStreamEvent::Start {
                id: "msg_test".to_string(),
                model: "test-model".to_string(),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );

        // Text delta — opens block 0
        let _ = stream_event_to_anthropic_events(
            NormalizedStreamEvent::Delta {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some("Let me run that.".to_string()),
                },
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert_eq!(block_state.active_index, Some(0));
        assert!(!block_state.active_is_tool);

        // Tool call starts — should close text block 0, open tool block 1
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::ToolCallDelta {
                index: 0,
                tool_call_index: 1,
                id: Some("call_xyz".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some("Bash".to_string()),
                    arguments: None,
                }),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert!(block_state.active_is_tool);
        assert!(block_state.active_index.is_some());
        // Should have ContentBlockStop for index 0, then ContentBlockStart for index 1
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStop { index: 0 }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStart { index: 1, .. }))
        );

        // End — should close tool block
        let tool_idx = block_state.active_index.unwrap();
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::End {
                finish_reason: FinishReason::ToolCalls,
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert_eq!(block_state.active_index, None);
        assert!(events.iter().any(
            |e| matches!(e, AnthropicStreamEvent::ContentBlockStop { index } if *index == tool_idx)
        ));
    }

    #[test]
    fn test_streaming_text_then_tool_at_same_openai_index() {
        // OpenAI sends text deltas at index 0, then tool_call at tool_call_index 0.
        // Anthropic indices must NOT collide — tool should get a unique index.
        use lunaroute_core::normalized::{Delta, FinishReason, FunctionCallDelta};

        let mut block_state = StreamBlockState::default();

        // Start
        let _ = stream_event_to_anthropic_events(
            NormalizedStreamEvent::Start {
                id: "msg_test".to_string(),
                model: "test-model".to_string(),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );

        // Text delta — gets Anthropic index 0
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::Delta {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some("Sure, let me check.".to_string()),
                },
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        let text_idx = block_state.active_index.unwrap();
        assert_eq!(text_idx, 0);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStart { index: 0, .. }))
        );

        // Tool call at OpenAI tool_call_index 0 — must get Anthropic index 1, NOT 0
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::ToolCallDelta {
                index: 0,
                tool_call_index: 0, // Same as text index!
                id: Some("call_abc".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some("Bash".to_string()),
                    arguments: None,
                }),
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        let tool_idx = block_state.active_index.unwrap();
        // Tool must have a DIFFERENT Anthropic index than text
        assert_ne!(tool_idx, text_idx);
        assert!(block_state.active_is_tool);
        // Should close text block at index 0
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStop { index: 0 }))
        );
        // Should open tool block at a new index (1)
        assert!(events
            .iter()
            .any(|e| matches!(e, AnthropicStreamEvent::ContentBlockStart { index, .. } if *index == tool_idx)));

        // End
        let events = stream_event_to_anthropic_events(
            NormalizedStreamEvent::End {
                finish_reason: FinishReason::ToolCalls,
            },
            "msg_test",
            "test-model",
            &mut block_state,
        );
        assert_eq!(block_state.active_index, None);
        assert!(events.iter().any(
            |e| matches!(e, AnthropicStreamEvent::ContentBlockStop { index } if *index == tool_idx)
        ));
    }
}
