package gollama

import (
	"testing"
	"time"
)

// Truncated must recognize an output-cap cutoff from either the normalized
// top-level StopReason (Anthropic "max_tokens") or the per-choice FinishReason
// (OpenAI "length"), and must not fire on a clean finish.
func TestClientSetTimeout(t *testing.T) {
	client := NewClient("https://example.invalid")
	if got, want := client.httpClient.Timeout, 300*time.Second; got != want {
		t.Fatalf("default timeout = %s, want %s", got, want)
	}

	client.SetTimeout(12 * time.Minute)
	if got, want := client.httpClient.Timeout, 12*time.Minute; got != want {
		t.Fatalf("configured timeout = %s, want %s", got, want)
	}

	client.SetTimeout(0)
	if got := client.httpClient.Timeout; got != 0 {
		t.Fatalf("disabled timeout = %s, want 0", got)
	}
}

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
