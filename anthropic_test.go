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
