package gollama

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// Client represents an Ollama API client
type Client struct {
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
}

// NewClient creates a new Ollama client
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		headers: make(map[string]string),
	}
}

func (c *Client) SetApiKey(k string) {
	c.SetHeader("x-api-key", k)
}

func (c *Client) SetHeader(k, v string) {
	c.headers[k] = v
}

// RequestOptions contains options for Generate requests
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

type ToolParam struct {
	Type     string        `json:"type"`
	Function *ToolFunction `json:"function,omitempty"`
}

type ToolFunction struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Parameters  ToolFunctionParams `json:"parameters"`
}

type ToolFunctionParams struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required"`
}

// Options contains model parameters
type Options struct {
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	MaxTokens   int     `json:"num_predict,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role             string     `json:"role"`
	Content          string     `json:"content,omitempty"`
	Thinking         string     `json:"thinking,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	Images           []string   `json:"images,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// MarshalJSON implements custom JSON marshaling for OpenAI ChatCompletion format
func (m Message) MarshalJSON() ([]byte, error) {
	type MessageAlias Message

	// If no images, use standard marshaling
	if len(m.Images) == 0 {
		alias := MessageAlias(m)
		return json.Marshal(alias)
	}

	// Create OpenAI-compatible content structure with images
	content := []map[string]interface{}{
		{
			"type": "text",
			"text": m.Content,
		},
	}

	// Add images in OpenAI format
	for _, img := range m.Images {
		content = append(content, map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:image/jpeg;base64,%s", img),
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

type ToolCall struct {
	Function ToolCallFunction `json:"function"`
	Type     string           `json:"type"`
	ID       string           `json:"id"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// GenerateResponse represents a response from the Ollama API generate endpoint
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

// ResponseMessage represents a response from the Ollama API chat endpoint
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

type ResponseMessageGenerate struct {
	Model     string      `json:"model"`
	Choices   []GenChoice `json:"choices"`
	CreatedAt string      `json:"created_at"`
	Done      bool        `json:"done"`
	Error     string      `json:"error,omitempty"`
}

type GenChoice struct {
	Index   int     `json:"index"`
	Message Message `json:"message"`
}

// Generate sends a completion request to the Ollama API
func (c *Client) Generate(opts RequestOptions) (*GenerateResponse, error) {
	// Set up request
	resp, err := c.prepareRequest(opts, "/api/generate")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Handle streaming if requested
	if opts.Stream {
		return c.handleGenerateStream(resp)
	}

	// Handle regular response
	return c.handleGenerateResponse(resp)
}

// HandleGenerateResponse processes a non-streaming generate response
func (c *Client) handleGenerateResponse(resp *http.Response) (*GenerateResponse, error) {
	decoder := json.NewDecoder(resp.Body)
	var response GenerateResponse
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &response, nil
}

// HandleGenerateStream processes a streaming generate response
func (c *Client) handleGenerateStream(resp *http.Response) (*GenerateResponse, error) {
	decoder := json.NewDecoder(resp.Body)
	var response GenerateResponse
	if decoder.More() {
		err := decoder.Decode(&response)
		if err != nil {
			return nil, fmt.Errorf("error decoding streaming response: %w", err)
		}
		return &response, nil
	}
	return nil, fmt.Errorf("empty response stream")
}

// Chat sends a chat completion request to the Ollama API
func (c *Client) Chat(opts RequestOptions) (*ResponseMessage, error) {
	// Set up request
	resp, err := c.prepareRequest(opts, "/api/chat")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Handle streaming if requested
	if opts.Stream {
		return c.handleChatStream(resp)
	}

	// Handle regular response
	return c.handleChatResponse(resp)
}

type ModelDesc struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Root   string `json:"root"`
}

type listModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelDesc `json:"data"`
}

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

// Chat sends a chat completion request to the Ollama API
func (c *Client) ChatCompletion(opts RequestOptions) (*ResponseMessageGenerate, error) {
	// Set up request
	resp, err := c.prepareRequest(opts, "/chat/completions")
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

func (c *Client) handleChatResponse(resp *http.Response) (*ResponseMessage, error) {
	decoder := json.NewDecoder(resp.Body)
	var response ResponseMessage
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &response, nil
}

func (c *Client) handleChatStream(resp *http.Response) (*ResponseMessage, error) {
	return nil, fmt.Errorf("not dealing with streams yet")
}

func (c *Client) prepareRequest(body any, endpoint string) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	req, err := http.NewRequest("POST", fmt.Sprintf("%s%s", c.baseURL, endpoint), bytes.NewBuffer(data))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API returned non-200 status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return resp, nil
}

func (c *Client) prepareGet(endpoint string) (*http.Response, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("%s%s", c.baseURL, endpoint), nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API returned non-200 status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return resp, nil
}

// AddImageFromFile encodes an image file as base64 and adds it to a message
func AddImageFromFile(filename string) (string, error) {
	// Read the file and encode as base64
	imageBytes, err := os.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("error reading image file: %w", err)
	}

	return base64.StdEncoding.EncodeToString(imageBytes), nil
}
