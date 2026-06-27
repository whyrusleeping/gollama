package gollama

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// TestBuildAnthropicRequest_ThinkingAndEffort verifies that opts.Thinking /
// Effort / ThinkingDisplay translate into the Anthropic request body, and that
// nothing is emitted when they're unset.
func TestBuildAnthropicRequest_ThinkingAndEffort(t *testing.T) {
	req, err := buildAnthropicRequest(RequestOptions{
		Model:           "claude-opus-4-8",
		Messages:        []Message{{Role: "user", Content: "hi"}},
		Thinking:        "adaptive",
		ThinkingDisplay: "summarized",
		Effort:          "xhigh",
	})
	if err != nil {
		t.Fatal(err)
	}
	if req.Thinking == nil || req.Thinking.Type != "adaptive" || req.Thinking.Display != "summarized" {
		t.Fatalf("thinking = %+v, want adaptive/summarized", req.Thinking)
	}
	if req.OutputConfig == nil || req.OutputConfig.Effort != "xhigh" {
		t.Fatalf("output_config = %+v, want effort=xhigh", req.OutputConfig)
	}

	// Serialized shape: budget_tokens must NOT appear (removed on current models).
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "budget_tokens") {
		t.Fatalf("request must not contain budget_tokens: %s", b)
	}
	if !strings.Contains(string(b), `"thinking":{"type":"adaptive"`) ||
		!strings.Contains(string(b), `"output_config":{"effort":"xhigh"}`) {
		t.Fatalf("serialized request missing thinking/effort: %s", b)
	}

	// Unset by default.
	req2, err := buildAnthropicRequest(RequestOptions{
		Model:    "claude-opus-4-8",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if req2.Thinking != nil || req2.OutputConfig != nil {
		t.Fatalf("expected no thinking/output_config by default; got %+v / %+v", req2.Thinking, req2.OutputConfig)
	}
}

// TestBuildAnthropicRequest_ReplaysThinkingBlocks verifies that an assistant
// turn's captured thinking blocks are re-emitted FIRST and verbatim (signature
// intact) — the round-trip Anthropic requires for tool-using conversations.
func TestBuildAnthropicRequest_ReplaysThinkingBlocks(t *testing.T) {
	req, err := buildAnthropicRequest(RequestOptions{
		Model: "claude-opus-4-8",
		Messages: []Message{
			{Role: "user", Content: "add 2 and 3"},
			{
				Role:    "assistant",
				Content: "I'll add them.",
				ThinkingBlocks: []ThinkingBlock{
					{Thinking: "2+3=5", Signature: "sig-abc"},
					{Redacted: "REDACTED-DATA"},
				},
				ToolCalls: []ToolCall{{
					ID:       "tu_1",
					Type:     "function",
					Function: ToolCallFunction{Name: "add", Arguments: `{"a":2,"b":3}`},
				}},
			},
			{Role: "tool", ToolCallID: "tu_1", Content: "5"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var asst *anthropicMessage
	for i := range req.Messages {
		if req.Messages[i].Role == "assistant" {
			asst = &req.Messages[i]
		}
	}
	if asst == nil || len(asst.Content) < 3 {
		t.Fatalf("assistant message malformed: %+v", asst)
	}
	// Block 0 must be the thinking block (thinking must come first), with signature.
	first, ok := asst.Content[0].(anthropicThinkingReqBlock)
	if !ok {
		t.Fatalf("first content block is %T, want thinking block first", asst.Content[0])
	}
	if first.Type != "thinking" || first.Thinking != "2+3=5" || first.Signature != "sig-abc" {
		t.Fatalf("thinking block = %+v", first)
	}
	// Block 1 must be the redacted thinking block.
	second, ok := asst.Content[1].(anthropicThinkingReqBlock)
	if !ok || second.Type != "redacted_thinking" || second.Data != "REDACTED-DATA" {
		t.Fatalf("second block = %+v, want redacted_thinking", asst.Content[1])
	}

	b, _ := json.Marshal(req)
	if !strings.Contains(string(b), `"signature":"sig-abc"`) ||
		!strings.Contains(string(b), `"type":"redacted_thinking"`) {
		t.Fatalf("serialized request missing replayed thinking: %s", b)
	}
}

// TestParseAnthropicResponse_Thinking verifies thinking and redacted_thinking
// blocks are parsed into Message.Thinking (text) and Message.ThinkingBlocks
// (verbatim, for replay), alongside text and tool_use.
func TestParseAnthropicResponse_Thinking(t *testing.T) {
	body := `{
	  "id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8",
	  "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":20},
	  "content":[
	    {"type":"thinking","thinking":"let me think","signature":"sig-xyz"},
	    {"type":"redacted_thinking","data":"OPAQUE"},
	    {"type":"text","text":"the answer is 5"},
	    {"type":"tool_use","id":"tu_1","name":"add","input":{"a":2,"b":3}}
	  ]}`
	resp := &http.Response{Body: io.NopCloser(strings.NewReader(body))}
	out, err := parseAnthropicResponse(resp)
	if err != nil {
		t.Fatal(err)
	}
	m := out.Choices[0].Message
	if m.Thinking != "let me think" {
		t.Errorf("thinking text = %q, want %q", m.Thinking, "let me think")
	}
	if len(m.ThinkingBlocks) != 2 {
		t.Fatalf("thinking blocks = %d, want 2", len(m.ThinkingBlocks))
	}
	if m.ThinkingBlocks[0].Thinking != "let me think" || m.ThinkingBlocks[0].Signature != "sig-xyz" {
		t.Errorf("block0 = %+v", m.ThinkingBlocks[0])
	}
	if m.ThinkingBlocks[1].Redacted != "OPAQUE" {
		t.Errorf("block1 redacted = %q, want OPAQUE", m.ThinkingBlocks[1].Redacted)
	}
	if m.Content != "the answer is 5" {
		t.Errorf("content = %q", m.Content)
	}
	if len(m.ToolCalls) != 1 || m.ToolCalls[0].Function.Name != "add" {
		t.Fatalf("tool calls = %+v", m.ToolCalls)
	}
}
