//! In-process (SDK) MCP servers.
//!
//! Instead of spawning a separate MCP server process, tools can be defined
//! directly in Rust and served from within the SDK. Such a server is declared
//! to the CLI as an `sdk`-type MCP server; the CLI then speaks JSON-RPC to it by
//! wrapping each request in an `mcp_message` control request. This module
//! implements that server side: it answers `initialize`, `tools/list`, and
//! `tools/call`, dispatching calls to the registered Rust handlers.
//!
//! Tools are exposed to the model as `mcp__<server>__<tool>`; add that name to
//! [`ClaudeOptions::allowed_tools`](crate::claude::ClaudeOptions) to let the
//! model call it.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Serialize;
use serde_json::{Value, json};

/// The MCP protocol version the SDK servers advertise.
const PROTOCOL_VERSION: &str = "2024-11-05";

/// The result of an MCP tool invocation.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ToolResult {
    /// Content blocks returned to the model (typically a single `text` block).
    pub content: Vec<Value>,
    /// Marks the call as failed; the model sees the content as an error.
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolResult {
    /// A successful text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![json!({ "type": "text", "text": text.into() })],
            is_error: None,
        }
    }

    /// An error text result.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![json!({ "type": "text", "text": text.into() })],
            is_error: Some(true),
        }
    }
}

/// A boxed, shareable async tool handler.
pub type ToolHandler =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = ToolResult> + Send>> + Send + Sync + 'static>;

/// A single tool exposed by an [`SdkMcpServer`].
#[derive(Clone)]
pub struct SdkMcpTool {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's arguments.
    pub input_schema: Value,
    handler: ToolHandler,
}

impl SdkMcpTool {
    /// Define a tool from an async handler.
    ///
    /// ```
    /// use agent_sdk::claude::{SdkMcpTool, ToolResult};
    /// use serde_json::json;
    ///
    /// let tool = SdkMcpTool::new(
    ///     "add",
    ///     "Add two numbers",
    ///     json!({ "type": "object", "properties": { "a": {"type": "number"}, "b": {"type": "number"} } }),
    ///     |args| async move {
    ///         let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
    ///         let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
    ///         ToolResult::text((a + b).to_string())
    ///     },
    /// );
    /// ```
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            handler: Arc::new(move |args| Box::pin(handler(args))),
        }
    }

    fn descriptor(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        })
    }
}

impl fmt::Debug for SdkMcpTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SdkMcpTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("input_schema", &self.input_schema)
            .finish_non_exhaustive()
    }
}

/// An in-process MCP server: a named bundle of [`SdkMcpTool`]s.
#[derive(Clone, Debug)]
pub struct SdkMcpServer {
    pub name: String,
    pub version: String,
    pub tools: Vec<SdkMcpTool>,
}

impl SdkMcpServer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "0.1.0".to_string(),
            tools: Vec::new(),
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Add a tool. Chainable.
    pub fn tool(mut self, tool: SdkMcpTool) -> Self {
        self.tools.push(tool);
        self
    }

    /// The `--mcp-config` entry announcing this as an `sdk`-type server. The CLI
    /// recognises the `sdk` type and routes its JSON-RPC over the control
    /// protocol rather than spawning a process.
    pub(crate) fn config_entry(&self) -> Value {
        json!({ "type": "sdk", "name": self.name })
    }

    /// Handle one JSON-RPC request from the CLI and produce the response.
    ///
    /// Returns `None` for notifications (messages without an `id`), which carry
    /// no JSON-RPC response.
    pub(crate) async fn handle_message(&self, message: Value) -> Option<Value> {
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();

        // Notifications have no id and expect no response.
        let id = message.get("id").cloned()?;

        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": self.name, "version": self.version },
            })),
            "tools/list" => Ok(json!({
                "tools": self.tools.iter().map(SdkMcpTool::descriptor).collect::<Vec<_>>(),
            })),
            "tools/call" => self.call_tool(message.get("params")).await,
            other => Err(JsonRpcError::method_not_found(other)),
        };

        Some(match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(error) => json!({ "jsonrpc": "2.0", "id": id, "error": error.to_value() }),
        })
    }

    async fn call_tool(&self, params: Option<&Value>) -> Result<Value, JsonRpcError> {
        let params = params.cloned().unwrap_or(Value::Null);
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| JsonRpcError::invalid_params("missing tool name"))?;

        let tool = self
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .ok_or_else(|| JsonRpcError::method_not_found(name))?;

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let result = (tool.handler)(arguments).await;

        Ok(serde_json::to_value(result).unwrap_or(Value::Null))
    }
}

struct JsonRpcError {
    code: i64,
    message: String,
}

impl JsonRpcError {
    fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
        }
    }

    fn invalid_params(message: &str) -> Self {
        Self {
            code: -32602,
            message: message.to_string(),
        }
    }

    fn to_value(&self) -> Value {
        json!({ "code": self.code, "message": self.message })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adder() -> SdkMcpServer {
        SdkMcpServer::new("calc").tool(SdkMcpTool::new(
            "add",
            "Add two numbers",
            json!({ "type": "object" }),
            |args| async move {
                let a = args.get("a").and_then(Value::as_i64).unwrap_or(0);
                let b = args.get("b").and_then(Value::as_i64).unwrap_or(0);
                ToolResult::text((a + b).to_string())
            },
        ))
    }

    #[tokio::test]
    async fn initialize_reports_server_info() {
        let response = adder()
            .handle_message(json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" }))
            .await
            .expect("response");
        assert_eq!(response["result"]["serverInfo"]["name"], "calc");
        assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn tools_list_describes_tools() {
        let response = adder()
            .handle_message(json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }))
            .await
            .expect("response");
        let tools = response["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "add");
    }

    #[tokio::test]
    async fn tools_call_dispatches_to_handler() {
        let response = adder()
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": { "name": "add", "arguments": { "a": 2, "b": 3 } },
            }))
            .await
            .expect("response");
        assert_eq!(response["result"]["content"][0]["text"], "5");
    }

    #[tokio::test]
    async fn unknown_tool_returns_method_not_found() {
        let response = adder()
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": { "name": "missing" },
            }))
            .await
            .expect("response");
        assert_eq!(response["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn notifications_have_no_response() {
        let response = adder()
            .handle_message(json!({ "method": "notifications/initialized" }))
            .await;
        assert!(response.is_none());
    }
}
