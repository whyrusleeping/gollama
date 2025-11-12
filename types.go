package gollama

import (
	"encoding/json"
)

// RequestOptions contains options for API requests.
// Used by Generate, Chat, and ChatCompletion methods.
type RequestOptions struct {
	Model      string      `json:"model"`
	Prompt     string      `json:"prompt,omitempty"`
	System     string      `json:"system,omitempty"`
	Context    []int       `json:"context,omitempty"`
	Format     string      `json:"format,omitempty"`
	Raw        bool        `json:"raw,omitempty"`
	Images     []string    `json:"images,omitempty"`
	Stream     bool        `json:"stream"`
	Messages   []Message   `json:"messages,omitempty"`
	Options    *Options    `json:"options,omitempty"`
	Think      bool        `json:"think"`
	Tools      []ToolParam `json:"tools,omitempty"`
	ToolChoice string      `json:"tool_choice,omitempty"`
}

// Options contains model parameters for controlling generation behavior.
type Options struct {
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	MaxTokens   int     `json:"num_predict,omitempty"`
}

// Message represents a chat message with support for text, images, and tool calls.
type Message struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	Thinking         string     `json:"thinking,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	Images           []string   `json:"images,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// MarshalJSON implements custom JSON marshaling supporting both OpenAI and Anthropic formats.
// When images are present, it uses Anthropic's structured content format which is more universal.
func (m Message) MarshalJSON() ([]byte, error) {
	type MessageAlias Message

	// If no images, use standard marshaling
	if len(m.Images) == 0 {
		alias := MessageAlias(m)
		return json.Marshal(alias)
	}

	// Create content structure with images
	// Use Anthropic format (type: "image") which is more universal
	// OpenAI endpoints will handle this, but Anthropic batch API requires it
	content := []map[string]interface{}{
		{
			"type": "text",
			"text": m.Content,
		},
	}

	// Add images in Anthropic format
	for _, img := range m.Images {
		content = append(content, map[string]interface{}{
			"type": "image",
			"source": map[string]string{
				"type":       "base64",
				"media_type": "image/jpeg",
				"data":       img,
			},
		})
	}

	// Create the final message structure
	result := map[string]interface{}{
		"role":    m.Role,
		"content": content,
	}

	// Add optional fields if present
	if m.Thinking != "" {
		result["thinking"] = m.Thinking
	}
	if m.ReasoningContent != "" {
		result["reasoning_content"] = m.ReasoningContent
	}
	if len(m.ToolCalls) > 0 {
		result["tool_calls"] = m.ToolCalls
	}
	if m.ToolCallID != "" {
		result["tool_call_id"] = m.ToolCallID
	}

	return json.Marshal(result)
}

// GenerateResponse represents a response from the Ollama /api/generate endpoint.
type GenerateResponse struct {
	Model              string `json:"model"`
	CreatedAt          string `json:"created_at"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	Context            []int  `json:"context,omitempty"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	Error              string `json:"error,omitempty"`
}

// ResponseMessage represents a response from the Ollama /api/chat endpoint.
type ResponseMessage struct {
	Model              string  `json:"model"`
	Message            Message `json:"message"`
	CreatedAt          string  `json:"created_at"`
	Done               bool    `json:"done"`
	Error              string  `json:"error,omitempty"`
	PromptEvalCount    int     `json:"prompt_eval_count"`
	PromptEvalDuration int64   `json:"prompt_eval_duration,omitempty"`
	EvalDuration       int64   `json:"eval_duration,omitempty"`
	EvalCount          int     `json:"eval_count,omitempty"`
}

// ResponseMessageGenerate represents a response from OpenAI-compatible /chat/completions endpoint.
type ResponseMessageGenerate struct {
	Model     string      `json:"model"`
	Choices   []GenChoice `json:"choices"`
	CreatedAt string      `json:"created_at"`
	Done      bool        `json:"done"`
	Error     string      `json:"error,omitempty"`
	Usage     Usage       `json:"usage"`
}

// GenChoice represents a single completion choice in OpenAI-compatible responses.
type GenChoice struct {
	Index   int     `json:"index"`
	Message Message `json:"message"`
}

// Usage represents token usage information in API responses.
type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details"`
}

// PromptTokensDetails provides detailed information about prompt token usage.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// ModelDesc represents a model description from the OpenAI-compatible /models endpoint.
type ModelDesc struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Root   string `json:"root"`
}

// listModelsResponse is an internal type for deserializing the /models endpoint response.
type listModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelDesc `json:"data"`
}
