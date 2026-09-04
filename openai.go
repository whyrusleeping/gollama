package gollama

import (
	"context"
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
// Generation parameters are top-level per the OpenAI spec; the nested Options field is
// kept for Ollama-compatible backends that expect it.
type openaiRequest struct {
	Model string      `json:"model"`
	Tools []ToolParam `json:"tools,omitempty"`
	// ReasoningEffort is the OpenAI-compatible reasoning knob
	// ("low"|"medium"|"high"|"xhigh", model-dependent). Set only for the OpenAI
	// backend from RequestOptions.Effort; never sent to Ollama.
	ReasoningEffort string    `json:"reasoning_effort,omitempty"`
	ToolChoice      string    `json:"tool_choice,omitempty"`
	Messages        []Message `json:"messages"`
	Stream          bool      `json:"stream,omitempty"`
	MaxTokens       int       `json:"max_tokens,omitempty"`
	Temperature     *float64  `json:"temperature,omitempty"`
	TopP            *float64  `json:"top_p,omitempty"`
	Options         *Options  `json:"options,omitempty"`
	// StreamOptions is set only when Stream is true. include_usage asks the
	// server to emit a final chunk carrying token usage (supported by OpenAI,
	// Ollama's /v1 endpoint, llama.cpp, and vLLM); without it many servers omit
	// usage entirely from a streamed response.
	StreamOptions *openaiStreamOptions `json:"stream_options,omitempty"`
	// Think is Ollama's on/off reasoning flag. Set only for the Ollama backend
	// (from RequestOptions.Think or a non-empty Thinking); Ollama exposes no
	// effort levels, so RequestOptions.Effort is deliberately dropped there.
	Think bool `json:"think,omitempty"`
}

// openaiStreamOptions is the stream_options object for streaming requests.
type openaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// mapOpenAIEffort translates a gollama Effort level into the value OpenAI's
// reasoning_effort field accepts. gollama levels are low|medium|high|xhigh|max;
// OpenAI documents none|minimal|low|medium|high|xhigh (model-dependent). "max"
// has no OpenAI equivalent, so it clamps to the highest expressible level,
// "xhigh"; anything else passes through unchanged.
func mapOpenAIEffort(effort string) string {
	if effort == "max" {
		return "xhigh"
	}
	return effort
}

// buildOpenAIRequest constructs the final request body for the OpenAI-compatible
// /chat/completions endpoint from opts. It performs system-message injection,
// tool-parameter normalization, top-level option promotion, per-backend reasoning
// translation (Ollama think bool vs OpenAI reasoning_effort), and the ExtraBody
// merge. The returned value is the exact body sent by both the non-streaming
// (ChatCompletion) and streaming (chatCompletionOpenAIStream) paths, so requests
// are byte-identical apart from opts.Stream and its stream_options. When
// opts.Stream is set, stream_options{include_usage:true} is added so the server
// emits a final usage chunk.
func (c *Client) buildOpenAIRequest(opts RequestOptions) (any, error) {
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

	// Normalize tool parameters for strict OpenAI-compatible servers (e.g. llama.cpp)
	// that reject null where an array is expected. Replace nil slices/maps with
	// empty ones so they serialize as [] / {} instead of null.
	tools := opts.Tools
	for i := range tools {
		if tools[i].Function == nil {
			continue
		}
		if tfp, ok := tools[i].Function.Parameters.(*ToolFunctionParams); ok {
			if tfp.Required == nil {
				tfp.Required = []string{}
			}
			if tfp.Properties == nil {
				tfp.Properties = map[string]any{}
			}
		} else if tfp, ok := tools[i].Function.Parameters.(ToolFunctionParams); ok {
			if tfp.Required == nil {
				tfp.Required = []string{}
			}
			if tfp.Properties == nil {
				tfp.Properties = map[string]any{}
			}
			tools[i].Function.Parameters = tfp
		}
	}

	// Build a clean request with field order optimized for prefix caching:
	// model -> tools (static) -> messages (dynamic)
	req := openaiRequest{
		Model:      opts.Model,
		Tools:      tools,
		ToolChoice: opts.ToolChoice,
		Messages:   messages,
		Stream:     opts.Stream,
	}
	// Only Ollama understands the nested options object; strict
	// OpenAI-compatible servers (e.g. Fireworks) reject unknown fields, so for
	// everything else the generation parameters ride solely as the promoted
	// top-level fields below.
	if c.Backend() == BackendOllama {
		req.Options = opts.Options
	}
	if opts.Stream {
		req.StreamOptions = &openaiStreamOptions{IncludeUsage: true}
	}

	// Promote Options to top-level fields for OpenAI-compatible backends
	if opts.Options != nil {
		if opts.Options.MaxTokens > 0 {
			req.MaxTokens = opts.Options.MaxTokens
		}
		if opts.Options.Temperature != 0 {
			req.Temperature = &opts.Options.Temperature
		}
		if opts.Options.TopP != 0 {
			req.TopP = &opts.Options.TopP
		}
	}

	// Translate reasoning controls per backend. Both backends flow through this
	// OpenAI-compatible path, but they express reasoning differently:
	//   - OpenAI: a reasoning_effort level (from Effort; "max" clamps to xhigh).
	//     There is no OpenAI request equivalent for Thinking — Effort is the knob.
	//   - Ollama: a think on/off bool only (no effort levels). It is enabled when
	//     Think is set or an adaptive Thinking is requested; Effort is ignored.
	switch c.Backend() {
	case BackendOllama:
		if opts.Think || opts.Thinking != "" {
			req.Think = true
		}
	default: // BackendOpenAI and other OpenAI-compatible endpoints
		if opts.Effort != "" {
			req.ReasoningEffort = mapOpenAIEffort(opts.Effort)
		}
	}

	// If ExtraBody is set, merge its keys into the request as top-level fields.
	if len(opts.ExtraBody) > 0 {
		raw, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("marshal request for extra body merge: %w", err)
		}
		var merged map[string]any
		if err := json.Unmarshal(raw, &merged); err != nil {
			return nil, fmt.Errorf("unmarshal request for extra body merge: %w", err)
		}
		for k, v := range opts.ExtraBody {
			merged[k] = v
		}
		return merged, nil
	}

	return req, nil
}

// ChatCompletion sends a chat completion request.
// If connected to Anthropic's API, uses the native /v1/messages endpoint with caching.
// Otherwise, uses the OpenAI-compatible /chat/completions endpoint.
// Returns a ResponseMessageGenerate with choices and usage information.
func (c *Client) ChatCompletion(opts RequestOptions) (*ResponseMessageGenerate, error) {
	return c.ChatCompletionCtx(context.Background(), opts)
}

// ChatCompletionCtx is ChatCompletion with caller-controlled cancellation and deadlines.
func (c *Client) ChatCompletionCtx(ctx context.Context, opts RequestOptions) (*ResponseMessageGenerate, error) {
	// Use AWS Bedrock endpoint
	if c.IsBedrockAPI() {
		return c.ChatCompletionBedrockCtx(ctx, opts)
	}

	// Use native Anthropic API for caching support
	if c.IsAnthropicAPI() {
		return c.ChatCompletionAnthropicCtx(ctx, opts)
	}

	body, err := c.buildOpenAIRequest(opts)
	if err != nil {
		return nil, err
	}

	// Set up request for OpenAI-compatible endpoint
	resp, err := c.prepareRequestCtx(ctx, body, "/chat/completions")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Handle regular response
	decoder := json.NewDecoder(resp.Body)
	var response ResponseMessageGenerate
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	// Normalize reasoning into the single Message.Thinking field. Ollama's /v1
	// endpoint returns reasoning text in message.reasoning; fold it into Thinking
	// (when Thinking is empty) so callers read one field regardless of backend,
	// matching the Anthropic path. Clear Reasoning afterward so assistant-turn
	// replay never re-emits a provider-specific reasoning key.
	if len(response.Choices) > 0 {
		msg := &response.Choices[0].Message
		if msg.Thinking == "" && msg.Reasoning != "" {
			msg.Thinking = msg.Reasoning
		}
		msg.Reasoning = ""
	}

	return &response, nil
}
