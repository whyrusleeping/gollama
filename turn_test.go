package gollama

import "testing"

// Truncated must recognize an output-cap cutoff from either the normalized
// top-level StopReason (Anthropic "max_tokens") or the per-choice FinishReason
// (OpenAI "length"), and must not fire on a clean finish.
func TestTruncated(t *testing.T) {
	cases := []struct {
		name string
		resp ResponseMessageGenerate
		want bool
	}{
		{"anthropic max_tokens", ResponseMessageGenerate{StopReason: "max_tokens"}, true},
		{"anthropic end_turn", ResponseMessageGenerate{StopReason: "end_turn"}, false},
		{"anthropic tool_use", ResponseMessageGenerate{StopReason: "tool_use"}, false},
		{"openai length", ResponseMessageGenerate{Choices: []GenChoice{{FinishReason: "length"}}}, true},
		{"openai stop", ResponseMessageGenerate{Choices: []GenChoice{{FinishReason: "stop"}}}, false},
		{"empty", ResponseMessageGenerate{}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.resp.Truncated(); got != c.want {
				t.Fatalf("Truncated() = %v, want %v", got, c.want)
			}
		})
	}
}
