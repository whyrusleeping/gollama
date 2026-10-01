package gollama

import (
	"strings"
	"testing"
)

// TestBuildAnthropicRequest_SystemRoleNormalization checks that role="system"
// messages anywhere in the input are pulled out into req.System rather than
// passed through to the /v1/messages endpoint (which rejects them with
// "Unexpected role 'system'").
func TestBuildAnthropicRequest_SystemRoleNormalization(t *testing.T) {
	cases := []struct {
		name        string
		opts        RequestOptions
		wantSystem  string
		wantNoSysIn []string // role values that must not appear in req.Messages
	}{
		{
			name: "system as first message",
			opts: RequestOptions{
				Model: "claude-sonnet-4-20250514",
				Messages: []Message{
					{Role: "system", Content: "you are helpful"},
					{Role: "user", Content: "hi"},
				},
			},
			wantSystem:  "you are helpful",
			wantNoSysIn: []string{"system"},
		},
		{
			name: "system role and explicit System: explicit wins, system msg dropped",
			opts: RequestOptions{
				Model:  "claude-sonnet-4-20250514",
				System: "explicit system",
				Messages: []Message{
					{Role: "system", Content: "stale system"},
					{Role: "user", Content: "hi"},
				},
			},
			wantSystem:  "explicit system",
			wantNoSysIn: []string{"system"},
		},
		{
			name: "system role mid-conversation",
			opts: RequestOptions{
				Model: "claude-sonnet-4-20250514",
				Messages: []Message{
					{Role: "user", Content: "hi"},
					{Role: "assistant", Content: "hello"},
					{Role: "system", Content: "remember to be brief"},
					{Role: "user", Content: "what's the weather?"},
				},
			},
			wantSystem:  "remember to be brief",
			wantNoSysIn: []string{"system"},
		},
		{
			name: "multiple system messages join",
			opts: RequestOptions{
				Model: "claude-sonnet-4-20250514",
				Messages: []Message{
					{Role: "system", Content: "first"},
					{Role: "system", Content: "second"},
					{Role: "user", Content: "hi"},
				},
			},
			wantSystem:  "first\n\nsecond",
			wantNoSysIn: []string{"system"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, err := buildAnthropicRequest(tc.opts)
			if err != nil {
				t.Fatalf("buildAnthropicRequest: %v", err)
			}

			var gotSystem string
			for _, sb := range req.System {
				if gotSystem != "" {
					gotSystem += "\n\n"
				}
				gotSystem += sb.Text
			}
			if !strings.Contains(gotSystem, tc.wantSystem) {
				t.Errorf("system block:\n  got:  %q\n  want substring: %q", gotSystem, tc.wantSystem)
			}

			for _, badRole := range tc.wantNoSysIn {
				for i, m := range req.Messages {
					if m.Role == badRole {
						t.Errorf("message %d has forbidden role %q (would be rejected by /v1/messages)", i, badRole)
					}
				}
			}
		})
	}
}

// toolResultIDs returns, for each converted message, the tool_use_ids of its
// tool_result blocks. Messages carrying no tool_result blocks yield a nil entry,
// which keeps the indexes lined up with req.Messages.
func toolResultIDs(msgs []anthropicMessage) [][]string {
	out := make([][]string, len(msgs))
	for i, m := range msgs {
		for _, block := range m.Content {
			if tr, ok := block.(anthropicToolResultBlock); ok {
				out[i] = append(out[i], tr.ToolUseID)
			}
		}
	}
	return out
}

func assistantWithToolCalls(ids ...string) Message {
	m := Message{Role: "assistant"}
	for _, id := range ids {
		m.ToolCalls = append(m.ToolCalls, ToolCall{
			Type:     "function",
			ID:       id,
			Function: ToolCallFunction{Name: "get_weather", Arguments: "{}"},
		})
	}
	return m
}

func toolResults(ids ...string) []Message {
	msgs := make([]Message, 0, len(ids))
	for _, id := range ids {
		msgs = append(msgs, Message{Role: "tool", ToolCallID: id, Content: "ok"})
	}
	return msgs
}

func concatMessages(groups ...[]Message) []Message {
	var out []Message
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

// TestBuildAnthropicRequest_ToolResultMergeAdjacency checks that consecutive
// tool results are merged into one user message only when they are actually
// adjacent in the source message list. A role="system" message is promoted to
// req.System and skipped during conversion; it must still break the run, or the
// tool results after it are appended to the user message belonging to an
// earlier assistant turn and the API rejects the request with
// "unexpected tool_use_id found in tool_result blocks".
func TestBuildAnthropicRequest_ToolResultMergeAdjacency(t *testing.T) {
	cases := []struct {
		name     string
		messages []Message
		// want lists the expected tool_use_ids of each converted message that
		// carries tool_result blocks, in order.
		want [][]string
	}{
		{
			name: "system between tool runs starts a new message",
			messages: concatMessages(
				[]Message{{Role: "user", Content: "weather?"}},
				[]Message{assistantWithToolCalls("toolu_A", "toolu_B", "toolu_C", "toolu_D")},
				toolResults("toolu_A", "toolu_B", "toolu_C", "toolu_D"),
				[]Message{{Role: "system", Content: "be brief"}},
				toolResults("toolu_ORPHAN"),
			),
			want: [][]string{
				{"toolu_A", "toolu_B", "toolu_C", "toolu_D"},
				{"toolu_ORPHAN"},
			},
		},
		{
			name: "system between tool runs with matching assistant turns",
			messages: concatMessages(
				[]Message{{Role: "user", Content: "weather?"}},
				[]Message{assistantWithToolCalls("toolu_A", "toolu_B")},
				toolResults("toolu_A", "toolu_B"),
				[]Message{{Role: "system", Content: "be brief"}},
				[]Message{assistantWithToolCalls("toolu_C", "toolu_D")},
				toolResults("toolu_C", "toolu_D"),
			),
			want: [][]string{
				{"toolu_A", "toolu_B"},
				{"toolu_C", "toolu_D"},
			},
		},
		{
			name: "contiguous tool run still merges",
			messages: concatMessages(
				[]Message{{Role: "user", Content: "weather?"}},
				[]Message{assistantWithToolCalls("toolu_A", "toolu_B", "toolu_C", "toolu_D")},
				toolResults("toolu_A", "toolu_B", "toolu_C", "toolu_D"),
			),
			want: [][]string{
				{"toolu_A", "toolu_B", "toolu_C", "toolu_D"},
			},
		},
		{
			name: "user message between tool runs starts a new message",
			messages: concatMessages(
				[]Message{{Role: "user", Content: "weather?"}},
				[]Message{assistantWithToolCalls("toolu_A")},
				toolResults("toolu_A"),
				[]Message{{Role: "user", Content: "and tomorrow?"}},
				[]Message{assistantWithToolCalls("toolu_B")},
				toolResults("toolu_B"),
			),
			want: [][]string{
				{"toolu_A"},
				{"toolu_B"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, err := buildAnthropicRequest(RequestOptions{
				Model:    "claude-sonnet-4-20250514",
				Messages: tc.messages,
			})
			if err != nil {
				t.Fatalf("buildAnthropicRequest: %v", err)
			}

			var got [][]string
			for _, ids := range toolResultIDs(req.Messages) {
				if len(ids) > 0 {
					got = append(got, ids)
				}
			}

			if len(got) != len(tc.want) {
				t.Fatalf("tool_result messages:\n  got:  %v\n  want: %v", got, tc.want)
			}
			for i := range tc.want {
				if strings.Join(got[i], ",") != strings.Join(tc.want[i], ",") {
					t.Errorf("tool_result message %d:\n  got:  %v\n  want: %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}
