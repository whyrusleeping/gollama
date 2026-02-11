package gollama

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ListModels retrieves the list of available models from the OpenAI-compatible /models endpoint.
func (c *Client) ListModels() ([]ModelDesc, error) {
	resp, err := c.prepareGet("/models")
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()

	var out listModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}

	return out.Data, nil
}

// openaiRequest is the request body for OpenAI-compatible /chat/completions endpoints.
// Field order is chosen to maximize prefix caching: model and tools (static) come before
// messages (dynamic), so the stable prefix is as long as possible.
type openaiRequest struct {
	Model      string      `json:"model"`
	Tools      []ToolParam `json:"tools,omitempty"`
	ToolChoice string      `json:"tool_choice,omitempty"`
	Messages   []Message   `json:"messages"`
	Stream     bool        `json:"stream,omitempty"`
	Options    *Options    `json:"options,omitempty"`
}

// ChatCompletion sends a chat completion request.
// If connected to Anthropic's API, uses the native /v1/messages endpoint with caching.
// Otherwise, uses the OpenAI-compatible /chat/completions endpoint.
// Returns a ResponseMessageGenerate with choices and usage information.
func (c *Client) ChatCompletion(opts RequestOptions) (*ResponseMessageGenerate, error) {
	// Use native Anthropic API for caching support
	if c.IsAnthropicAPI() {
		return c.ChatCompletionAnthropic(opts)
	}

	// For OpenAI-compatible APIs, inject system prompt as a system-role message
	// at the front of the messages array. SystemBlocks takes priority over System string.
	messages := opts.Messages
	if len(opts.SystemBlocks) > 0 {
		var sb strings.Builder
		for i, block := range opts.SystemBlocks {
			if i > 0 {
				sb.WriteString("\n\n")
			}
			sb.WriteString(block.Text)
		}
		messages = append([]Message{{Role: "system", Content: sb.String()}}, messages...)
	} else if opts.System != "" {
		messages = append([]Message{{Role: "system", Content: opts.System}}, messages...)
	}

	// Build a clean request with field order optimized for prefix caching:
	// model -> tools (static) -> messages (dynamic)
	req := openaiRequest{
		Model:      opts.Model,
		Tools:      opts.Tools,
		ToolChoice: opts.ToolChoice,
		Messages:   messages,
		Stream:     opts.Stream,
		Options:    opts.Options,
	}

	// Set up request for OpenAI-compatible endpoint
	resp, err := c.prepareRequest(req, "/chat/completions")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Handle streaming if requested
	if opts.Stream {
		return nil, fmt.Errorf("not doing streaming")
	}

	// Handle regular response
	decoder := json.NewDecoder(resp.Body)
	var response ResponseMessageGenerate
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &response, nil
}
