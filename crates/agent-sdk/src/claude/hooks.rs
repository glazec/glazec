//! Hook callbacks.
//!
//! Claude Code can call back into the SDK at well-defined points in a turn
//! (before/after a tool runs, when the model stops, …). The Python SDK exposes
//! these as "hooks"; this module brings the same capability to the Rust SDK.
//!
//! A hook is an async callback registered against a [`HookEvent`] and an
//! optional matcher (a tool-name pattern, only meaningful for tool events).
//! During [`Claude::connect`](crate::claude::Claude::connect) the registered
//! callbacks are announced to the CLI in the `initialize` control request; the
//! CLI then issues `hook_callback` control requests which the SDK routes back to
//! the matching callback and answers with the callback's [`HookOutput`].

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

/// The points at which Claude Code can invoke a hook.
///
/// The string form (see [`HookEvent::as_str`]) is the identifier the CLI uses on
/// the wire; it matches the event names accepted by Claude Code settings hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    Notification,
    Stop,
    SubagentStop,
    PreCompact,
    SessionStart,
    SessionEnd,
}

impl HookEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::Notification => "Notification",
            Self::Stop => "Stop",
            Self::SubagentStop => "SubagentStop",
            Self::PreCompact => "PreCompact",
            Self::SessionStart => "SessionStart",
            Self::SessionEnd => "SessionEnd",
        }
    }
}

impl fmt::Display for HookEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Input handed to a hook callback when the CLI invokes it.
///
/// The strongly-typed fields cover the common cases; [`HookInput::raw`] always
/// carries the full payload so callbacks can read fields not surfaced here.
#[derive(Debug, Clone, PartialEq)]
pub struct HookInput {
    /// The event that triggered this callback (e.g. `PreToolUse`).
    pub hook_event_name: Option<String>,
    /// For tool events, the tool being used (e.g. `Bash`).
    pub tool_name: Option<String>,
    /// For tool events, the tool's input arguments.
    pub tool_input: Option<Value>,
    /// For `PostToolUse`, the tool's response.
    pub tool_response: Option<Value>,
    /// The `tool_use_id` associated with the call, when present.
    pub tool_use_id: Option<String>,
    /// The complete, unmodified payload from the CLI.
    pub raw: Value,
}

impl HookInput {
    pub(crate) fn from_request(request: &Value, tool_use_id: Option<String>) -> Self {
        let input = request.get("input").cloned().unwrap_or(Value::Null);

        let field = |key: &str| {
            input
                .get(key)
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        };

        Self {
            hook_event_name: field("hook_event_name"),
            tool_name: field("tool_name"),
            tool_input: input.get("tool_input").cloned(),
            tool_response: input.get("tool_response").cloned(),
            tool_use_id: tool_use_id.or_else(|| {
                input
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            }),
            raw: input,
        }
    }
}

/// Permission decision a `PreToolUse` hook can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionDecision {
    Allow,
    Deny,
    Ask,
}

/// The value a hook callback returns to influence the turn.
///
/// `HookOutput::default()` is an empty, no-op response (the turn proceeds
/// unchanged) — the common case for observe-only hooks. Use the builder methods
/// to block a tool call, inject a system message, and so on.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct HookOutput {
    /// Whether Claude should continue after the hook runs. `Some(false)` halts
    /// the turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#continue: Option<bool>,
    /// Reason surfaced when `continue` is `false`.
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Suppress the hook's stdout from the transcript.
    #[serde(rename = "suppressOutput", skip_serializing_if = "Option::is_none")]
    pub suppress_output: Option<bool>,
    /// `approve` / `block` decision for events that support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    /// Human-readable reason accompanying `decision`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Additional context injected as a system message.
    #[serde(rename = "systemMessage", skip_serializing_if = "Option::is_none")]
    pub system_message: Option<String>,
    /// Event-specific structured output (e.g. a `PreToolUse` permission
    /// decision). Passed through to the CLI verbatim.
    #[serde(rename = "hookSpecificOutput", skip_serializing_if = "Option::is_none")]
    pub hook_specific_output: Option<Value>,
}

impl HookOutput {
    /// An empty, observe-only response.
    pub fn new() -> Self {
        Self::default()
    }

    /// Block the action and tell Claude why.
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            decision: Some("block".to_string()),
            reason: Some(reason.into()),
            ..Self::default()
        }
    }

    /// Halt the entire turn with a reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self {
            r#continue: Some(false),
            stop_reason: Some(reason.into()),
            ..Self::default()
        }
    }

    /// Return a `PreToolUse` permission decision.
    pub fn permission(decision: PermissionDecision, reason: impl Into<String>) -> Self {
        Self {
            hook_specific_output: Some(serde_json::json!({
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason.into(),
            })),
            ..Self::default()
        }
    }

    /// Attach a system message to an otherwise-passthrough response.
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.system_message = Some(message.into());
        self
    }

    pub(crate) fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// A boxed, shareable async hook callback.
pub type HookCallback = Arc<
    dyn Fn(HookInput) -> Pin<Box<dyn Future<Output = HookOutput> + Send>> + Send + Sync + 'static,
>;

/// Adapt an async closure into a [`HookCallback`].
///
/// ```
/// use agent_sdk::claude::{hook, HookOutput};
///
/// let cb = hook(|input| async move {
///     println!("tool: {:?}", input.tool_name);
///     HookOutput::new()
/// });
/// ```
pub fn hook<F, Fut>(callback: F) -> HookCallback
where
    F: Fn(HookInput) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = HookOutput> + Send + 'static,
{
    Arc::new(move |input| Box::pin(callback(input)))
}

/// A set of hook callbacks gated by an optional tool-name matcher.
#[derive(Clone)]
pub struct HookMatcher {
    /// Tool-name pattern (CLI semantics). `None` matches every tool/event.
    pub matcher: Option<String>,
    /// Callbacks invoked when this matcher fires.
    pub callbacks: Vec<HookCallback>,
}

impl HookMatcher {
    /// Match a specific tool name (or any pattern the CLI understands).
    pub fn new(matcher: impl Into<String>, callbacks: Vec<HookCallback>) -> Self {
        Self {
            matcher: Some(matcher.into()),
            callbacks,
        }
    }

    /// Match every invocation of the event regardless of tool.
    pub fn any(callbacks: Vec<HookCallback>) -> Self {
        Self {
            matcher: None,
            callbacks,
        }
    }
}

impl fmt::Debug for HookMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HookMatcher")
            .field("matcher", &self.matcher)
            .field(
                "callbacks",
                &format_args!("[{} callback(s)]", self.callbacks.len()),
            )
            .finish()
    }
}

/// Hook configuration: which matchers fire for which events.
///
/// Stored on [`ClaudeOptions`](crate::claude::ClaudeOptions). Build it with
/// [`HookRegistry::on`] then hand it to `connect`.
#[derive(Debug, Clone, Default)]
pub struct HookRegistry {
    pub(crate) events: BTreeMap<HookEvent, Vec<HookMatcher>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Register a matcher for an event. Chainable.
    pub fn on(mut self, event: HookEvent, matcher: HookMatcher) -> Self {
        self.events.entry(event).or_default().push(matcher);
        self
    }

    /// Resolve the registry into the `initialize` payload the CLI expects and a
    /// `callback_id -> callback` map the client uses to route `hook_callback`
    /// requests. Callback ids are assigned deterministically (`hook_0`, …).
    pub(crate) fn build(&self) -> (Value, BTreeMap<String, HookCallback>) {
        let mut registry = BTreeMap::new();
        let mut config = serde_json::Map::new();
        let mut next_id = 0usize;

        for (event, matchers) in &self.events {
            let mut entries = Vec::with_capacity(matchers.len());

            for matcher in matchers {
                let mut callback_ids = Vec::with_capacity(matcher.callbacks.len());

                for callback in &matcher.callbacks {
                    let id = format!("hook_{next_id}");
                    next_id += 1;
                    registry.insert(id.clone(), Arc::clone(callback));
                    callback_ids.push(Value::String(id));
                }

                let mut entry = serde_json::Map::new();
                if let Some(pattern) = &matcher.matcher {
                    entry.insert("matcher".to_string(), Value::String(pattern.clone()));
                }
                entry.insert("hookCallbackIds".to_string(), Value::Array(callback_ids));
                entries.push(Value::Object(entry));
            }

            config.insert(event.as_str().to_string(), Value::Array(entries));
        }

        (Value::Object(config), registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_assigns_sequential_callback_ids_and_camelcase_payload() {
        let registry = HookRegistry::new()
            .on(
                HookEvent::PreToolUse,
                HookMatcher::new("Bash", vec![hook(|_| async { HookOutput::block("no") })]),
            )
            .on(
                HookEvent::Stop,
                HookMatcher::any(vec![hook(|_| async { HookOutput::new() })]),
            );

        let (payload, callbacks) = registry.build();

        assert_eq!(callbacks.len(), 2);
        assert!(callbacks.contains_key("hook_0"));
        assert!(callbacks.contains_key("hook_1"));

        assert_eq!(
            payload,
            serde_json::json!({
                "PreToolUse": [ { "matcher": "Bash", "hookCallbackIds": ["hook_0"] } ],
                "Stop": [ { "hookCallbackIds": ["hook_1"] } ],
            })
        );
    }

    #[test]
    fn hook_output_serializes_only_set_fields() {
        assert_eq!(HookOutput::new().to_value(), serde_json::json!({}));
        assert_eq!(
            HookOutput::block("nope").to_value(),
            serde_json::json!({ "decision": "block", "reason": "nope" })
        );
        assert_eq!(
            HookOutput::permission(PermissionDecision::Deny, "blocked").to_value(),
            serde_json::json!({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "blocked",
                }
            })
        );
    }

    #[test]
    fn hook_input_extracts_typed_fields() {
        let request = serde_json::json!({
            "input": {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": { "command": "ls" },
            }
        });
        let input = HookInput::from_request(&request, Some("toolu_9".to_string()));
        assert_eq!(input.hook_event_name.as_deref(), Some("PreToolUse"));
        assert_eq!(input.tool_name.as_deref(), Some("Bash"));
        assert_eq!(input.tool_use_id.as_deref(), Some("toolu_9"));
        assert_eq!(
            input.tool_input,
            Some(serde_json::json!({ "command": "ls" }))
        );
    }
}
