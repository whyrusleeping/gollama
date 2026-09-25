package gollama

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// anthropicStreamEvent is the envelope for a single Anthropic Messages streaming
// SSE event. Only the fields consumed by the assembler are modeled; unknown
// fields (and unknown event types) are ignored for forward compatibility.
type anthropicStreamEvent struct {
	Type string `json:"type"`

	// message_start / message_delta
	Message *anthropicResponse    `json:"message,omitempty"`
	Delta   *anthropicStreamDelta `json:"delta,omitempty"`
	Usage   *anthropicUsage       `json:"usage,omitempty"`

	// content_block_start / content_block_delta / content_block_stop
	Index        int                    `json:"index"`
	ContentBlock *anthropicContentBlock `json:"content_block,omitempty"`

	// error
	Error *anthropicStreamError `json:"error,omitempty"`
}

// anthropicStreamDelta carries the payload of a content_block_delta (text /
// thinking / signature / input_json partials) or a message_delta (stop reason).
type anthropicStreamDelta struct {
	Type string `json:"type"`

	// text_delta
	Text string `json:"text,omitempty"`
	// thinking_delta
	Thinking string `json:"thinking,omitempty"`
	// signature_delta
	Signature string `json:"signature,omitempty"`
	// input_json_delta
	PartialJSON string `json:"partial_json,omitempty"`

	// message_delta
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

type anthropicStreamError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// chatCompletionAnthropicStream runs a native Anthropic Messages turn with
// server-sent-event streaming enabled. It fires onDelta with the full
// accumulated assistant text (a snapshot, not an increment) each time a
// text_delta arrives, then assembles the complete response from the stream and
// returns it via convertAnthropicResponse — the same converter used by the
// non-streaming path, which guarantees the final message is byte-equivalent to
// what ChatCompletionAnthropic would have returned for the same response.
func (c *Client) chatCompletionAnthropicStream(ctx context.Context, opts RequestOptions, onDelta func(text string)) (*ResponseMessageGenerate, error) {
	req, err := buildAnthropicRequest(opts)
	if err != nil {
		return nil, err
	}
	req.Stream = true

	// The native Anthropic API requires an anthropic-version header. Default it
	// here so callers don't have to; a caller that set one explicitly wins.
	if _, ok := c.headers["anthropic-version"]; !ok {
		c.SetHeader("anthropic-version", "2023-06-01")
	}

	// prepareRequest handles pre-stream retryable statuses (429/503/529) exactly
	// like the non-streaming path; a 200 returns the open SSE body to consume.
	resp, err := c.prepareRequestCtx(ctx, req, c.anthropicEndpoint("/messages"))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	assembled, err := assembleAnthropicStream(resp.Body, onDelta)
	if err != nil {
		return nil, err
	}
	return convertAnthropicResponse(assembled), nil
}

// assembleAnthropicStream reads an Anthropic Messages SSE body and reconstructs
// the equivalent anthropicResponse, invoking onDelta with the accumulated text
// snapshot on each text_delta. It reads with a bufio.Reader (not a Scanner)
// because individual data lines can exceed the default scanner buffer.
func assembleAnthropicStream(body io.Reader, onDelta func(text string)) (*anthropicResponse, error) {
	reader := bufio.NewReader(body)

	assembled := &anthropicResponse{Type: "message", Role: "assistant"}
	// Content blocks assembled strictly in index order.
	byIndex := map[int]*anthropicContentBlock{}
	var order []int
	// Accumulated raw partial_json per tool_use block index.
	toolJSON := map[int]*strings.Builder{}
	// Running concatenation of all text-block content, for onDelta snapshots.
	var textAccum strings.Builder
	var completed bool

	blockAt := func(idx int) *anthropicContentBlock {
		b, ok := byIndex[idx]
		if !ok {
			b = &anthropicContentBlock{}
			byIndex[idx] = b
			order = append(order, idx)
		}
		return b
	}

	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			// We only care about data lines; the event: line is advisory
			// because each data payload already carries its own "type".
			if data, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(data)
				if data == "" {
					continue
				}
				var ev anthropicStreamEvent
				if uerr := json.Unmarshal([]byte(data), &ev); uerr != nil {
					return nil, fmt.Errorf("error decoding Anthropic stream event: %w", uerr)
				}

				switch ev.Type {
				case "message_start":
					if ev.Message != nil {
						assembled.ID = ev.Message.ID
						assembled.Model = ev.Message.Model
						if ev.Message.Role != "" {
							assembled.Role = ev.Message.Role
						}
						assembled.Usage.InputTokens = ev.Message.Usage.InputTokens
						assembled.Usage.CacheCreationInputTokens = ev.Message.Usage.CacheCreationInputTokens
						assembled.Usage.CacheReadInputTokens = ev.Message.Usage.CacheReadInputTokens
						if ev.Message.Usage.OutputTokens > 0 {
							assembled.Usage.OutputTokens = ev.Message.Usage.OutputTokens
						}
					}
				case "content_block_start":
					b := blockAt(ev.Index)
					if ev.ContentBlock != nil {
						*b = *ev.ContentBlock
					}
					if b.Type == "tool_use" {
						toolJSON[ev.Index] = &strings.Builder{}
					}
				case "content_block_delta":
					b := blockAt(ev.Index)
					if ev.Delta != nil {
						switch ev.Delta.Type {
						case "text_delta":
							b.Text += ev.Delta.Text
							textAccum.WriteString(ev.Delta.Text)
							if onDelta != nil {
								onDelta(textAccum.String())
							}
						case "thinking_delta":
							b.Thinking += ev.Delta.Thinking
						case "signature_delta":
							b.Signature += ev.Delta.Signature
						case "input_json_delta":
							sb := toolJSON[ev.Index]
							if sb == nil {
								sb = &strings.Builder{}
								toolJSON[ev.Index] = sb
							}
							sb.WriteString(ev.Delta.PartialJSON)
						}
					}
				case "content_block_stop":
					b := blockAt(ev.Index)
					if b.Type == "tool_use" {
						if sb := toolJSON[ev.Index]; sb != nil && sb.Len() > 0 {
							var input any
							if uerr := json.Unmarshal([]byte(sb.String()), &input); uerr != nil {
								return nil, fmt.Errorf("error decoding tool_use input json: %w", uerr)
							}
							b.Input = input
						}
					}
				case "message_delta":
					if ev.Delta != nil {
						if ev.Delta.StopReason != "" {
							assembled.StopReason = ev.Delta.StopReason
						}
						if ev.Delta.StopSequence != nil {
							assembled.StopSequence = ev.Delta.StopSequence
						}
					}
					if ev.Usage != nil && ev.Usage.OutputTokens > 0 {
						assembled.Usage.OutputTokens = ev.Usage.OutputTokens
					}
				case "message_stop":
					completed = true
				case "error":
					if ev.Error != nil {
						return nil, fmt.Errorf("anthropic stream error (%s): %s", ev.Error.Type, ev.Error.Message)
					}
					return nil, fmt.Errorf("anthropic stream error")
				case "ping":
					// keep-alive; ignore
				default:
					// Unknown event type: ignore for forward compatibility.
				}
			}
		}

		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("error reading Anthropic stream: %w", err)
		}
	}

	if !completed {
		return nil, fmt.Errorf("anthropic stream ended before message_stop: %w", io.ErrUnexpectedEOF)
	}

	// Assemble content blocks in index order.
	for _, idx := range order {
		assembled.Content = append(assembled.Content, *byIndex[idx])
	}
	return assembled, nil
}
