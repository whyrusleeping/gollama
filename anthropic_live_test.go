package gollama

import (
	"os"
	"strings"
	"testing"
)

// liveAnthropicClient builds a client against the real Anthropic API, skipping
// the test when no key is available. Set ANTHROPIC_API_KEY to run.
func liveAnthropicClient(t *testing.T) *Client {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping live Anthropic test")
	}
	c := NewClient("https://api.anthropic.com")
	c.SetAnthropicMode(true)
	c.SetAPIKey(key)
	return c
}

const liveThinkingModel = "claude-opus-4-8"

// TestAnthropicThinkingLive_Reasoning checks that adaptive thinking + a
// summarized display returns parsed thinking blocks with text and a signature.
func TestAnthropicThinkingLive_Reasoning(t *testing.T) {
	c := liveAnthropicClient(t)
	resp, err := c.Turn(RequestOptions{
		Model:           liveThinkingModel,
		System:          "You are precise. Reason carefully before answering.",
		Messages:        []Message{{Role: "user", Content: "How many times does the letter r appear in 'strawberry'? Think it through, then give the number."}},
		Thinking:        "adaptive",
		ThinkingDisplay: "summarized",
		Effort:          "high",
		Options:         &Options{MaxTokens: 4096},
	})
	if err != nil {
		t.Fatalf("turn: %v", err)
	}
	m := resp.Choices[0].Message
	t.Logf("thinking_blocks=%d thinking_text_len=%d answer=%q", len(m.ThinkingBlocks), len(m.Thinking), m.Content)
	if len(m.ThinkingBlocks) == 0 {
		t.Fatalf("expected thinking blocks with adaptive thinking on")
	}
	if m.Thinking == "" {
		t.Errorf("expected summarized thinking text (display=summarized)")
	}
	if m.ThinkingBlocks[0].Signature == "" {
		t.Errorf("expected a signature on the thinking block (needed for replay)")
	}
	if !strings.Contains(m.Content, "3") {
		t.Errorf("expected the answer to contain 3; got %q", m.Content)
	}
}

// TestAnthropicThinkingLive_ToolRoundTrip is the important one: it runs a
// tool-using turn with thinking enabled, then replays the assistant turn (which
// carries thinking blocks) plus the tool result. This must NOT 400 — proving the
// thinking blocks are echoed back with valid signatures.
func TestAnthropicThinkingLive_ToolRoundTrip(t *testing.T) {
	c := liveAnthropicClient(t)
	tools := []ToolParam{{
		Type: "function",
		Function: &ToolFunction{
			Name:        "add",
			Description: "Add two integers and return their sum.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"a": map[string]any{"type": "integer"},
					"b": map[string]any{"type": "integer"},
				},
				"required": []string{"a", "b"},
			},
		},
	}}

	sys := "You are a careful assistant. When asked to add numbers, use the add tool."
	// Phrased to induce reasoning before the tool call, so turn 1 produces a
	// signed thinking block — that's what makes the turn-2 replay a real test of
	// the signature round-trip (not just a thinking-enabled request that 200s).
	msgs := []Message{{Role: "user", Content: "I have 1234 marbles and someone gives me 5678 more. Reason step by step about the total first, then call the add tool to verify, and finally state the total in a sentence."}}

	opts := func(ms []Message) RequestOptions {
		return RequestOptions{
			Model:           liveThinkingModel,
			System:          sys,
			Messages:        ms,
			Tools:           tools,
			Thinking:        "adaptive",
			ThinkingDisplay: "summarized",
			Effort:          "high",
			Options:         &Options{MaxTokens: 8192},
		}
	}

	resp, err := c.Turn(opts(msgs))
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	m := resp.Choices[0].Message
	t.Logf("turn1 thinking_blocks=%d tool_calls=%d text=%q", len(m.ThinkingBlocks), len(m.ToolCalls), m.Content)
	if len(m.ToolCalls) == 0 {
		t.Fatalf("expected a tool call on turn 1")
	}

	// Replay: assistant turn (with thinking blocks) + the tool result.
	tc := m.ToolCalls[0]
	msgs = append(msgs, m, Message{Role: "tool", ToolCallID: tc.ID, Content: "6912"})

	resp2, err := c.Turn(opts(msgs))
	if err != nil {
		// A 400 here would mean the thinking-block replay is broken.
		t.Fatalf("turn 2 (round-trip with thinking blocks): %v", err)
	}
	final := resp2.Choices[0].Message
	t.Logf("turn2 thinking_blocks=%d text=%q", len(final.ThinkingBlocks), final.Content)
	if !strings.Contains(strings.ReplaceAll(final.Content, ",", ""), "6912") {
		t.Errorf("expected final answer to mention 6912; got %q", final.Content)
	}
}
