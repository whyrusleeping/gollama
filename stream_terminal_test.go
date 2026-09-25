package gollama

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func TestOpenAIStreamNeedsTerminalSignal(t *testing.T) {
	partial := `data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}` + "\n\n"
	for _, tc := range []struct {
		name, body string
		complete   bool
	}{
		{"partial EOF", partial, false},
		{"finish reason", partial + `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n", true},
		{"done sentinel", partial + "data: [DONE]\n\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := assembleOpenAIStream(strings.NewReader(tc.body), nil)
			if tc.complete {
				if err != nil || resp == nil {
					t.Fatalf("complete stream: resp=%v err=%v", resp, err)
				}
			} else if resp != nil || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("truncated stream: resp=%v err=%v", resp, err)
			}
		})
	}
}

func TestAnthropicStreamNeedsMessageStop(t *testing.T) {
	partial := `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}` + "\n\n"
	for _, tc := range []struct {
		name, body string
		complete   bool
	}{
		{"partial EOF", partial, false},
		{"stop reason without message stop", partial + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}` + "\n\n", false},
		{"message stop", partial + `data: {"type":"message_stop"}` + "\n\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := assembleAnthropicStream(strings.NewReader(tc.body), nil)
			if tc.complete {
				if err != nil || resp == nil {
					t.Fatalf("complete stream: resp=%v err=%v", resp, err)
				}
			} else if resp != nil || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("truncated stream: resp=%v err=%v", resp, err)
			}
		})
	}
}
