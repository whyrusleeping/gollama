package gollama

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// openaiStreamChunk models a single chat.completion.chunk from an
// OpenAI-compatible /chat/completions streaming response. Only the fields the
// assembler consumes are modeled; unknown fields are ignored for forward
// compatibility. A final chunk may carry usage with an empty choices array.
type openaiStreamChunk struct {
	Model   string               `json:"model"`
	Choices []openaiStreamChoice `json:"choices"`
	Usage   *Usage               `json:"usage,omitempty"`
	// error is tolerated as a mid-stream object on some servers.
	Error *openaiStreamError `json:"error,omitempty"`
}

type openaiStreamChoice struct {
	Index        int               `json:"index"`
	Delta        openaiStreamDelta `json:"delta"`
	FinishReason string            `json:"finish_reason,omitempty"`
}

// openaiStreamDelta is the incremental payload of a streamed choice. content
// concatenates to the assistant text; reasoning / reasoning_content carry
// provider reasoning; tool_calls arrive fragmented and are keyed by index.
type openaiStreamDelta struct {
	Role             string                 `json:"role,omitempty"`
	Content          string                 `json:"content,omitempty"`
	Thinking         string                 `json:"thinking,omitempty"`
	Reasoning        string                 `json:"reasoning,omitempty"`
	ReasoningContent string                 `json:"reasoning_content,omitempty"`
	ToolCalls        []openaiStreamToolCall `json:"tool_calls,omitempty"`
}

// openaiStreamToolCall is a tool-call fragment. On the first fragment for a
// given index, id / type / function.name are present; function.arguments is a
// raw JSON string that concatenates across subsequent fragments.
type openaiStreamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

type openaiStreamError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// chatCompletionOpenAIStream runs a single OpenAI-compatible /chat/completions
// turn with server-sent-event streaming enabled. It fires onDelta with the full
// accumulated assistant text (a snapshot, not an increment) each time a content
// delta arrives, then assembles the complete response and normalizes it into the
// exact ResponseMessageGenerate that ChatCompletion would have returned for the
// equivalent non-streaming response — so the final message is byte-equivalent
// across the streaming and non-streaming paths. Reasoning deltas are accumulated
// but not delivered to onDelta (matching the Anthropic path: thinking is not
// streamed to onDelta).
func (c *Client) chatCompletionOpenAIStream(ctx context.Context, opts RequestOptions, onDelta func(text string)) (*ResponseMessageGenerate, error) {
	opts.Stream = true
	body, err := c.buildOpenAIRequest(opts)
	if err != nil {
		return nil, err
	}

	// prepareRequest handles pre-stream retryable statuses (429/503/529) exactly
	// like the non-streaming path; a 200 returns the open SSE body to consume.
	resp, err := c.prepareRequestCtx(ctx, body, "/chat/completions")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return assembleOpenAIStream(resp.Body, onDelta)
}

// assembleOpenAIStream reads an OpenAI-compatible chat-completions SSE body and
// reconstructs the equivalent ResponseMessageGenerate, invoking onDelta with the
// accumulated assistant-text snapshot on each content delta. It reads with a
// bufio.Reader (not a Scanner) because individual data lines can be long.
func assembleOpenAIStream(body io.Reader, onDelta func(text string)) (*ResponseMessageGenerate, error) {
	reader := bufio.NewReader(body)

	var (
		model            string
		role             string
		finishReason     string
		choiceIndex      int
		usage            Usage
		haveChoice       bool
		completed        bool
		contentAccum     strings.Builder
		thinkingAccum    strings.Builder
		reasoningAccum   strings.Builder
		reasoningContent strings.Builder
	)
	// Tool-call fragments accumulated in first-seen index order.
	toolByIndex := map[int]*ToolCall{}
	toolArgs := map[int]*strings.Builder{}
	var toolOrder []int

	for {
		line, rerr := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			if data, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(data)
				if data == "" {
					// event separator or comment; nothing to decode.
				} else if data == "[DONE]" {
					completed = true
					break
				} else {
					var chunk openaiStreamChunk
					if uerr := json.Unmarshal([]byte(data), &chunk); uerr != nil {
						return nil, fmt.Errorf("error decoding OpenAI stream chunk: %w", uerr)
					}
					if chunk.Error != nil {
						if chunk.Error.Type != "" {
							return nil, fmt.Errorf("openai stream error (%s): %s", chunk.Error.Type, chunk.Error.Message)
						}
						return nil, fmt.Errorf("openai stream error: %s", chunk.Error.Message)
					}
					if chunk.Model != "" {
						model = chunk.Model
					}
					if chunk.Usage != nil {
						usage = *chunk.Usage
					}
					for _, ch := range chunk.Choices {
						haveChoice = true
						choiceIndex = ch.Index
						if ch.FinishReason != "" {
							finishReason = ch.FinishReason
							completed = true
						}
						d := ch.Delta
						if d.Role != "" {
							role = d.Role
						}
						if d.Thinking != "" {
							thinkingAccum.WriteString(d.Thinking)
						}
						if d.Reasoning != "" {
							reasoningAccum.WriteString(d.Reasoning)
						}
						if d.ReasoningContent != "" {
							reasoningContent.WriteString(d.ReasoningContent)
						}
						if d.Content != "" {
							contentAccum.WriteString(d.Content)
							if onDelta != nil {
								onDelta(contentAccum.String())
							}
						}
						for _, tc := range d.ToolCalls {
							idx := tc.Index
							call, ok := toolByIndex[idx]
							if !ok {
								call = &ToolCall{}
								toolByIndex[idx] = call
								toolArgs[idx] = &strings.Builder{}
								toolOrder = append(toolOrder, idx)
							}
							if tc.ID != "" {
								call.ID = tc.ID
							}
							if tc.Type != "" {
								call.Type = tc.Type
							}
							if tc.Function.Name != "" {
								call.Function.Name = tc.Function.Name
							}
							if tc.Function.Arguments != "" {
								toolArgs[idx].WriteString(tc.Function.Arguments)
							}
						}
					}
				}
			}
		}

		if rerr != nil {
			if rerr == io.EOF {
				break
			}
			return nil, fmt.Errorf("error reading OpenAI stream: %w", rerr)
		}
	}

	if !completed {
		return nil, fmt.Errorf("openai stream ended before completion: %w", io.ErrUnexpectedEOF)
	}

	// If the stream produced no choices at all, still return a well-formed
	// (empty-choices) response carrying model and usage.
	response := &ResponseMessageGenerate{Model: model, Usage: usage}
	if !haveChoice && len(toolOrder) == 0 {
		return response, nil
	}

	msg := Message{
		Role:             role,
		Content:          contentAccum.String(),
		Thinking:         thinkingAccum.String(),
		Reasoning:        reasoningAccum.String(),
		ReasoningContent: reasoningContent.String(),
	}
	for _, idx := range toolOrder {
		call := toolByIndex[idx]
		call.Function.Arguments = toolArgs[idx].String()
		msg.ToolCalls = append(msg.ToolCalls, *call)
	}

	// Normalize reasoning into the single Message.Thinking field, exactly as
	// ChatCompletion does after decoding a non-streaming response.
	if msg.Thinking == "" && msg.Reasoning != "" {
		msg.Thinking = msg.Reasoning
	}
	msg.Reasoning = ""

	response.Choices = []GenChoice{{
		Index:        choiceIndex,
		Message:      msg,
		FinishReason: finishReason,
	}}
	return response, nil
}
