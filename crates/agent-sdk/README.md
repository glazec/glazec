# agent-sdk

A Rust SDK and client for agent runtimes. Currently only Claude Code is
supported.

## Claude Code

Rust client for [Claude Code](https://code.claude.com/docs/en/cli-reference).

Other SDKs:
- [Python SDK](https://github.com/anthropics/claude-agent-sdk-python).

### Long-lived agent runtime

The SDK turns Claude Code into a long-lived agent runtime. Instead of
launching separate `claude -p "…"` invocations and stitching sessions back
together with `--resume`, `Claude::connect` spawns a single subprocess that
stays alive across turns. Call `query` or `send_user_text` as many times as
you need — the underlying process, conversation history, and tool state
persist for the lifetime of the connection. This makes it straightforward to
build multi-turn agents, orchestration loops, and interactive applications on
top of Claude Code.

### What it provides

- `Claude::connect(options)` then `query` / `send_user_text` for sessions;
  call `shutdown` when done
- typed control helpers (`set_model`, `set_permission_mode`, `interrupt`,
  `rewind_files`, `get_mcp_status`, …)
- typed `ClaudeMessage` parsing for `assistant`, `user`, `system`, `result`,
  `stream_event`, and control protocol frames
- in-process MCP servers — define tools in Rust and serve them over the
  control protocol (no separate process)
- hook callbacks — observe and steer the agent at `PreToolUse`, `PostToolUse`,
  and other events
- low-level `Stdio` transport for raw newline-delimited JSON access

### Usage

```rust
use agent_sdk::claude::{Claude, ClaudeMessage, ClaudeOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = ClaudeOptions {
        model: Some("claude-sonnet-4-5".to_string()),
        ..ClaudeOptions::default()
    };
    let mut client = Claude::connect(options).await?;
    let response = client.query("Summarize this repo").await?;
    client.shutdown().await?;

    for message in response.messages {
        if let ClaudeMessage::Assistant(message) = message
            && let Some(text) = message.text()
        {
            println!("{text}");
        }
    }

    Ok(())
}
```

### In-process MCP servers

Define tools directly in Rust; they are exposed to the model as
`mcp__<server>__<tool>`. The CLI reaches them over the control protocol — no
extra process is spawned.

```rust
use agent_sdk::claude::{ClaudeOptions, SdkMcpServer, SdkMcpTool, ToolResult};
use serde_json::{Value, json};

let server = SdkMcpServer::new("calc").tool(SdkMcpTool::new(
    "add",
    "Add two numbers",
    json!({ "type": "object", "properties": { "a": { "type": "number" }, "b": { "type": "number" } } }),
    |args| async move {
        let a = args.get("a").and_then(Value::as_f64).unwrap_or(0.0);
        let b = args.get("b").and_then(Value::as_f64).unwrap_or(0.0);
        ToolResult::text((a + b).to_string())
    },
));

let options = ClaudeOptions {
    mcp_servers: vec![server],
    allowed_tools: vec!["mcp__calc__add".to_string()],
    ..ClaudeOptions::default()
};
```

The internal turn loop (`query` / `receive_response`) answers the CLI's
`mcp_message` requests automatically. When driving the stream by hand with
`next_message`, forward each `ClaudeMessage::ControlRequest` to
`Claude::handle_control_request`.

### Hook callbacks

Register async callbacks against hook events. A callback receives a `HookInput`
and returns a `HookOutput` — empty to observe, or built to block a tool call,
inject a system message, or return a permission decision.

```rust
use agent_sdk::claude::{ClaudeOptions, HookEvent, HookMatcher, HookOutput, HookRegistry, hook};

let hooks = HookRegistry::new().on(
    HookEvent::PreToolUse,
    HookMatcher::new(
        "Bash",
        vec![hook(|input| async move {
            eprintln!("about to run: {:?}", input.tool_name);
            HookOutput::new()
        })],
    ),
);

let options = ClaudeOptions { hooks, ..ClaudeOptions::default() };
```

Registered hooks are announced to the CLI during `initialize`; the SDK routes
each `hook_callback` request to the matching callback and replies with its
output.

### Example

`examples/claude_code.rs` covers both one-shot and interactive modes. When no
prompt is given, it starts an interactive REPL. Pass `--demo-tools` to register
an in-process MCP server (`mcp__demo__add`) plus a `PreToolUse` logging hook:

```bash
cargo run --example claude_code -- "What is 2 + 2?"
cargo run --example claude_code -- --model claude-sonnet-4-5 "Summarize this repo"
cargo run --example claude_code -- --cli-path /path/to/claude "Hello"
cargo run --example claude_code -- --demo-tools "Use add to sum 2 and 3"
cargo run --example claude_code --                              # REPL
cargo run --example claude_code -- --model claude-sonnet-4-5    # REPL with model
```

### Notes

- The SDK shells out to the external `claude` executable; it does not bundle
  Claude Code.
- Additional agent runtimes may be added in the future behind the same SDK
  surface.
