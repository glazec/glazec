pub mod hooks;
pub mod mcp;
pub mod messages;
pub mod options;

use std::collections::{BTreeMap, VecDeque};

use serde_json::{Value, json};
use thiserror::Error;

pub use self::hooks::{
    HookCallback, HookEvent, HookInput, HookMatcher, HookOutput, HookRegistry, PermissionDecision,
    hook,
};
pub use self::mcp::{SdkMcpServer, SdkMcpTool, ToolHandler, ToolResult};
use self::messages::parse_message;
pub use self::messages::{
    AssistantMessage, ClaudeMessage, ContentBlock, ControlRequestMessage, ControlResponseMessage,
    MessageContent, ResultMessage, StreamEventMessage, SystemMessage, TextBlock, ThinkingBlock,
    ToolResultBlock, ToolUseBlock, TurnResponse, UserMessage,
};
pub use self::options::{ClaudeOptions, PermissionMode, ToolConfig};
use crate::stdio::{Stdio, StdioError};

const CONTROL_REQUEST_ERROR: &str = "unsupported control request in cli-agent::Claude";

pub struct Claude {
    stdio: Stdio,
    request_counter: u64,
    pending_messages: VecDeque<ClaudeMessage>,
    server_info: Option<Value>,
    /// `callback_id -> callback` for hooks announced during `initialize`.
    hook_callbacks: BTreeMap<String, HookCallback>,
    /// In-process MCP servers, keyed by server name.
    mcp_servers: BTreeMap<String, SdkMcpServer>,
}

/// Literal tag stamped onto outbound `user` frames. The CLI requires the field
/// to be present on stream-json input but does not route on its value — the
/// session a subprocess belongs to is fixed by `--session-id`/`--resume` at
/// launch time. Anthropic's Python SDK hardcodes the same value.
const USER_FRAME_SESSION_TAG: &str = "default";

#[derive(Debug, Error)]
pub enum ClaudeError {
    #[error(transparent)]
    Stdio(#[from] StdioError),

    #[error("Claude CLI not found in PATH; install `claude` or set ClaudeOptions::cli_path")]
    CliNotFound,

    #[error("control request failed: {0}")]
    ControlRequest(String),
}

impl Claude {
    pub async fn connect(options: ClaudeOptions) -> Result<Self, ClaudeError> {
        let cli_path = options
            .resolve_cli_path()
            .map_err(|_| ClaudeError::CliNotFound)?;

        // Resolve hook and MCP registries before spawning so the subprocess can
        // be told, during `initialize`, which hook callbacks exist.
        let (mut hooks_payload, hook_callbacks) = options.hooks.build();
        if options.hooks.is_empty() {
            hooks_payload = Value::Null;
        }
        let mcp_servers = options
            .mcp_servers
            .iter()
            .map(|server| (server.name.clone(), server.clone()))
            .collect();

        let stdio = Stdio::spawn(cli_path, options.build_args(), options.stdio_options())?;

        let mut client = Self {
            stdio,
            request_counter: 0,
            pending_messages: VecDeque::new(),
            server_info: None,
            hook_callbacks,
            mcp_servers,
        };

        client.initialize(hooks_payload).await?;
        Ok(client)
    }

    pub fn pid(&self) -> Option<u32> {
        self.stdio.pid()
    }

    pub fn server_info(&self) -> Option<&Value> {
        self.server_info.as_ref()
    }

    async fn initialize(&mut self, hooks: Value) -> Result<(), ClaudeError> {
        let server_info = self
            .send_control_request(json!({
                "subtype": "initialize",
                "hooks": hooks,
            }))
            .await?;
        self.server_info = Some(server_info);
        Ok(())
    }

    pub async fn get_mcp_status(&mut self) -> Result<Value, ClaudeError> {
        self.send_control_request(json!({
            "subtype": "mcp_status",
        }))
        .await
    }

    pub async fn set_model(&mut self, model: impl Into<String>) -> Result<Value, ClaudeError> {
        self.send_control_request(json!({
            "subtype": "set_model",
            "model": model.into(),
        }))
        .await
    }

    pub async fn set_permission_mode(
        &mut self,
        permission_mode: PermissionMode,
    ) -> Result<Value, ClaudeError> {
        self.send_control_request(json!({
            "subtype": "set_permission_mode",
            "mode": permission_mode.as_cli_arg(),
        }))
        .await
    }

    pub async fn interrupt(&mut self) -> Result<Value, ClaudeError> {
        self.send_control_request(json!({
            "subtype": "interrupt",
        }))
        .await
    }

    pub async fn rewind_files(
        &mut self,
        user_message_id: impl Into<String>,
    ) -> Result<Value, ClaudeError> {
        self.send_control_request(json!({
            "subtype": "rewind_files",
            "user_message_id": user_message_id.into(),
        }))
        .await
    }

    pub async fn query(&mut self, prompt: impl Into<String>) -> Result<TurnResponse, ClaudeError> {
        self.send_user_text(prompt).await?;
        self.receive_response().await
    }

    pub async fn send_user_text(&mut self, text: impl Into<String>) -> Result<(), ClaudeError> {
        self.stdio
            .send_json(&json!({
                "type": "user",
                "message": {
                    "role": "user",
                    "content": text.into(),
                },
                "parent_tool_use_id": Value::Null,
                "session_id": USER_FRAME_SESSION_TAG,
            }))
            .await?;
        Ok(())
    }

    async fn receive_response(&mut self) -> Result<TurnResponse, ClaudeError> {
        let mut messages = Vec::new();

        loop {
            let message = self.next_message().await?;

            if let ClaudeMessage::ControlRequest(control_request) = &message {
                self.handle_control_request(control_request).await?;
                continue;
            }

            let is_result = matches!(message, ClaudeMessage::Result(_));
            messages.push(message);

            if is_result {
                return Ok(TurnResponse { messages });
            }
        }
    }

    pub async fn next_message(&mut self) -> Result<ClaudeMessage, ClaudeError> {
        if let Some(message) = self.pending_messages.pop_front() {
            return Ok(message);
        }

        self.read_message_from_stdio().await
    }

    /// Dispatch an incoming `control_request` from the CLI.
    ///
    /// `hook_callback` and `mcp_message` requests are routed to the registered
    /// hook/MCP handlers and answered with a `control_response`. Any other
    /// subtype is rejected as unsupported. The internal turn loops call this
    /// automatically; callers driving the stream by hand (via [`next_message`])
    /// should call it for each [`ClaudeMessage::ControlRequest`] they receive.
    ///
    /// [`next_message`]: Self::next_message
    pub async fn handle_control_request(
        &mut self,
        request: &ControlRequestMessage,
    ) -> Result<(), ClaudeError> {
        let Some(request_id) = request.request_id.clone() else {
            return Ok(());
        };

        match request.subtype.as_deref() {
            Some("hook_callback") => self.handle_hook_callback(&request_id, request).await,
            Some("mcp_message") => self.handle_mcp_message(&request_id, request).await,
            _ => {
                self.respond_control_request_error(&request_id, CONTROL_REQUEST_ERROR)
                    .await
            }
        }
    }

    async fn handle_hook_callback(
        &mut self,
        request_id: &str,
        request: &ControlRequestMessage,
    ) -> Result<(), ClaudeError> {
        let payload = request.request.as_ref();

        let Some(callback_id) = payload
            .and_then(|value| value.get("callback_id"))
            .and_then(Value::as_str)
        else {
            return self
                .respond_control_request_error(request_id, "hook_callback missing callback_id")
                .await;
        };

        let Some(callback) = self.hook_callbacks.get(callback_id).cloned() else {
            let error = format!("unknown hook callback: {callback_id}");
            return self.respond_control_request_error(request_id, &error).await;
        };

        let tool_use_id = payload
            .and_then(|value| value.get("tool_use_id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let input = HookInput::from_request(payload.unwrap_or(&Value::Null), tool_use_id);

        let output = callback(input).await;
        self.respond_control_request_success(request_id, output.to_value())
            .await
    }

    async fn handle_mcp_message(
        &mut self,
        request_id: &str,
        request: &ControlRequestMessage,
    ) -> Result<(), ClaudeError> {
        let payload = request.request.as_ref();

        let Some(server_name) = payload
            .and_then(|value| value.get("server_name"))
            .and_then(Value::as_str)
        else {
            return self
                .respond_control_request_error(request_id, "mcp_message missing server_name")
                .await;
        };

        let Some(server) = self.mcp_servers.get(server_name).cloned() else {
            let error = format!("unknown mcp server: {server_name}");
            return self.respond_control_request_error(request_id, &error).await;
        };

        let message = payload
            .and_then(|value| value.get("message"))
            .cloned()
            .unwrap_or(Value::Null);
        let mcp_response = server.handle_message(message).await.unwrap_or(Value::Null);

        self.respond_control_request_success(request_id, json!({ "mcp_response": mcp_response }))
            .await
    }

    pub async fn respond_control_request_error(
        &mut self,
        request_id: &str,
        error: &str,
    ) -> Result<(), ClaudeError> {
        self.stdio
            .send_json(&json!({
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": request_id,
                    "error": error,
                }
            }))
            .await?;

        Ok(())
    }

    pub async fn respond_control_request_success(
        &mut self,
        request_id: &str,
        response: Value,
    ) -> Result<(), ClaudeError> {
        self.stdio
            .send_json(&json!({
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": response,
                }
            }))
            .await?;

        Ok(())
    }

    pub async fn shutdown(&mut self) -> Result<(), ClaudeError> {
        self.stdio.shutdown().await?;
        Ok(())
    }

    async fn send_control_request(&mut self, request: Value) -> Result<Value, ClaudeError> {
        let request_id = self.next_request_id();

        self.stdio
            .send_json(&json!({
                "type": "control_request",
                "request_id": request_id,
                "request": request,
            }))
            .await?;

        loop {
            match self.read_message_from_stdio().await? {
                ClaudeMessage::ControlRequest(message) => {
                    self.handle_control_request(&message).await?;
                }
                ClaudeMessage::ControlResponse(response) => {
                    let Some(response_request_id) = response.request_id.as_deref() else {
                        self.pending_messages
                            .push_back(ClaudeMessage::ControlResponse(response));
                        continue;
                    };

                    if response_request_id != request_id {
                        self.pending_messages
                            .push_back(ClaudeMessage::ControlResponse(response));
                        continue;
                    }

                    match response.subtype.as_deref() {
                        Some("success") => return Ok(response.response.unwrap_or(Value::Null)),
                        Some("error") => {
                            let error = response
                                .error
                                .unwrap_or_else(|| "unknown control request error".to_string());
                            return Err(ClaudeError::ControlRequest(error));
                        }
                        _ => {
                            return Err(ClaudeError::ControlRequest(
                                "unknown control response subtype".to_string(),
                            ));
                        }
                    }
                }
                message => {
                    self.pending_messages.push_back(message);
                }
            }
        }
    }

    async fn read_message_from_stdio(&mut self) -> Result<ClaudeMessage, ClaudeError> {
        let raw = self.stdio.read_json::<Value>().await?;
        Ok(parse_message(raw))
    }

    fn next_request_id(&mut self) -> String {
        self.request_counter += 1;
        format!("req_{}", self.request_counter)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn query_collects_messages_and_result() {
        use std::os::unix::fs::PermissionsExt;

        let script_path = temp_script_path("query");
        let script = r#"#!/bin/sh
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"control_response","response":{"subtype":"success","request_id":"req_1","response":{"session_id":"session-123","ready":true}}}'
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"assistant","message":{"model":"claude-sonnet-4-5","content":[{"type":"text","text":"Hello from Claude"}]}}'
printf '%s\n' '{"type":"result","subtype":"success","duration_ms":100.0,"duration_api_ms":50.0,"is_error":false,"num_turns":1,"session_id":"session-123","total_cost_usd":0.001,"result":"done"}'
"#;

        fs::write(&script_path, script).expect("write fake claude script");

        let mut permissions = fs::metadata(&script_path)
            .expect("read script metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions).expect("chmod fake claude script");

        let mut client = connect_fake(ClaudeOptions {
            cli_path: Some(script_path.clone()),
            model: Some("claude-sonnet-4-5".to_string()),
            ..ClaudeOptions::default()
        })
        .await;

        let response = client.query("Say hello").await.expect("query succeeds");
        client.shutdown().await.expect("shutdown");

        assert_eq!(
            response.messages,
            vec![
                ClaudeMessage::Assistant(AssistantMessage {
                    content: vec![ContentBlock::Text(TextBlock {
                        text: "Hello from Claude".to_string(),
                        raw: json!({
                            "type": "text",
                            "text": "Hello from Claude"
                        }),
                    })],
                    model: Some("claude-sonnet-4-5".to_string()),
                    parent_tool_use_id: None,
                    error: None,
                    raw: json!({
                        "type": "assistant",
                        "message": {
                            "model": "claude-sonnet-4-5",
                            "content": [
                                { "type": "text", "text": "Hello from Claude" }
                            ]
                        }
                    }),
                }),
                ClaudeMessage::Result(ResultMessage {
                    subtype: Some("success".to_string()),
                    duration_ms: Some(100.0),
                    duration_api_ms: Some(50.0),
                    is_error: Some(false),
                    num_turns: Some(1),
                    session_id: Some("session-123".to_string()),
                    total_cost_usd: Some(0.001),
                    usage: None,
                    result: Some(json!("done")),
                    structured_output: None,
                    raw: json!({
                        "type":"result",
                        "subtype":"success",
                        "duration_ms":100.0,
                        "duration_api_ms":50.0,
                        "is_error":false,
                        "num_turns":1,
                        "session_id":"session-123",
                        "total_cost_usd":0.001,
                        "result":"done"
                    }),
                }),
            ]
        );
        assert_eq!(
            response.assistant_text(),
            Some("Hello from Claude".to_string())
        );
        assert_eq!(
            response
                .result()
                .and_then(|result| result.session_id.as_deref()),
            Some("session-123")
        );

        let _ = fs::remove_file(&script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn control_request_buffers_interleaved_non_control_messages() {
        use std::os::unix::fs::PermissionsExt;

        let script_path = temp_script_path("interleaved-control");
        let script = r#"#!/bin/sh
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"text","text":"queued assistant"}]}}'
printf '%s\n' '{"type":"control_response","response":{"subtype":"success","request_id":"req_1","response":{"ok":true}}}'
"#;

        fs::write(&script_path, script).expect("write fake claude script");

        let mut permissions = fs::metadata(&script_path)
            .expect("read script metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions).expect("chmod fake claude script");

        let mut client = connect_fake(ClaudeOptions {
            cli_path: Some(script_path.clone()),
            ..ClaudeOptions::default()
        })
        .await;

        assert_eq!(
            client.server_info().expect("initialized"),
            &json!({ "ok": true })
        );

        match client.next_message().await.expect("next buffered message") {
            ClaudeMessage::Assistant(message) => {
                assert_eq!(message.text().as_deref(), Some("queued assistant"));
            }
            other => panic!("expected buffered assistant message, got {other:?}"),
        }

        client.shutdown().await.expect("shutdown");
        let _ = fs::remove_file(&script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hook_callback_is_dispatched_and_answered() {
        use std::os::unix::fs::PermissionsExt;

        // The fake CLI: ack initialize, read the user line, emit a PreToolUse
        // hook_callback, read the SDK's control_response back, echo it inside an
        // assistant message so the test can assert the SDK answered correctly,
        // then finish the turn.
        let script_path = temp_script_path("hook-callback");
        let script = r#"#!/bin/sh
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"control_response","response":{"subtype":"success","request_id":"req_1","response":{"ready":true}}}'
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"control_request","request_id":"cli_1","request":{"subtype":"hook_callback","callback_id":"hook_0","tool_use_id":"toolu_7","input":{"hook_event_name":"PreToolUse","tool_name":"Bash"}}}'
IFS= read -r reply || exit 0
printf '%s\n' "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":$(printf '%s' "$reply" | sed 's/\\/\\\\/g; s/\"/\\\"/g' | sed 's/^/\"/; s/$/\"/')}]}}"
printf '%s\n' '{"type":"result","subtype":"success","duration_ms":1.0,"duration_api_ms":1.0,"is_error":false,"num_turns":1,"session_id":"s","total_cost_usd":0.0,"result":"ok"}'
"#;

        fs::write(&script_path, script).expect("write fake claude script");
        let mut permissions = fs::metadata(&script_path)
            .expect("read script metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions).expect("chmod fake claude script");

        let hooks = HookRegistry::new().on(
            HookEvent::PreToolUse,
            HookMatcher::new(
                "Bash",
                vec![hook(|input: HookInput| async move {
                    assert_eq!(input.tool_name.as_deref(), Some("Bash"));
                    assert_eq!(input.tool_use_id.as_deref(), Some("toolu_7"));
                    HookOutput::block("not allowed")
                })],
            ),
        );

        let mut client = connect_fake(ClaudeOptions {
            cli_path: Some(script_path.clone()),
            hooks,
            ..ClaudeOptions::default()
        })
        .await;

        let response = client.query("run ls").await.expect("query succeeds");
        client.shutdown().await.expect("shutdown");

        let echoed = response
            .assistant_text()
            .expect("assistant echoed the control_response");
        let parsed: Value = serde_json::from_str(&echoed).expect("control_response is valid json");
        assert_eq!(parsed["type"], "control_response");
        assert_eq!(parsed["response"]["subtype"], "success");
        assert_eq!(parsed["response"]["request_id"], "cli_1");
        assert_eq!(parsed["response"]["response"]["decision"], "block");
        assert_eq!(parsed["response"]["response"]["reason"], "not allowed");

        let _ = fs::remove_file(&script_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_tool_call_is_dispatched_and_answered() {
        use std::os::unix::fs::PermissionsExt;

        let script_path = temp_script_path("mcp-message");
        let script = r#"#!/bin/sh
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"control_response","response":{"subtype":"success","request_id":"req_1","response":{"ready":true}}}'
IFS= read -r _ || exit 0
printf '%s\n' '{"type":"control_request","request_id":"cli_9","request":{"subtype":"mcp_message","server_name":"calc","message":{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3}}}}}'
IFS= read -r reply || exit 0
printf '%s\n' "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":$(printf '%s' "$reply" | sed 's/\\/\\\\/g; s/\"/\\\"/g' | sed 's/^/\"/; s/$/\"/')}]}}"
printf '%s\n' '{"type":"result","subtype":"success","duration_ms":1.0,"duration_api_ms":1.0,"is_error":false,"num_turns":1,"session_id":"s","total_cost_usd":0.0,"result":"ok"}'
"#;

        fs::write(&script_path, script).expect("write fake claude script");
        let mut permissions = fs::metadata(&script_path)
            .expect("read script metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script_path, permissions).expect("chmod fake claude script");

        let server = SdkMcpServer::new("calc").tool(SdkMcpTool::new(
            "add",
            "Add two numbers",
            json!({ "type": "object" }),
            |args| async move {
                let a = args.get("a").and_then(Value::as_i64).unwrap_or(0);
                let b = args.get("b").and_then(Value::as_i64).unwrap_or(0);
                ToolResult::text((a + b).to_string())
            },
        ));

        let mut client = connect_fake(ClaudeOptions {
            cli_path: Some(script_path.clone()),
            mcp_servers: vec![server],
            ..ClaudeOptions::default()
        })
        .await;

        let response = client.query("add 2 and 3").await.expect("query succeeds");
        client.shutdown().await.expect("shutdown");

        let echoed = response
            .assistant_text()
            .expect("assistant echoed the control_response");
        let parsed: Value = serde_json::from_str(&echoed).expect("control_response is valid json");
        assert_eq!(parsed["response"]["subtype"], "success");
        assert_eq!(parsed["response"]["request_id"], "cli_9");
        let mcp_response = &parsed["response"]["response"]["mcp_response"];
        assert_eq!(mcp_response["id"], 1);
        assert_eq!(mcp_response["result"]["content"][0]["text"], "5");

        let _ = fs::remove_file(&script_path);
    }

    #[cfg(unix)]
    fn temp_script_path(prefix: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("unix epoch")
            .as_nanos();
        let pid = std::process::id();
        std::env::temp_dir().join(format!("cli-agent-{prefix}-{pid}-{now}.sh"))
    }

    /// Connect to a fake-CLI script, retrying on `ETXTBSY`.
    ///
    /// Tests run in parallel threads in one process. When one thread has just
    /// written an executable script, that writable fd can briefly leak into a
    /// `fork()` issued by another thread spawning its own script, making
    /// `execve` fail with `ETXTBSY` (errno 26). The condition is transient — the
    /// leaked fd is closed (CLOEXEC) once the other child execs — so a short
    /// retry resolves it. Real sessions spawn the installed `claude` binary and
    /// never hit this.
    #[cfg(unix)]
    async fn connect_fake(options: ClaudeOptions) -> Claude {
        use std::time::Duration;

        for _ in 0..100 {
            match Claude::connect(options.clone()).await {
                Ok(client) => return client,
                Err(ClaudeError::Stdio(StdioError::Spawn { source, .. }))
                    if source.raw_os_error() == Some(26) =>
                {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("connect fake claude: {error:?}"),
            }
        }

        panic!("connect fake claude: still ETXTBSY after retries");
    }
}
