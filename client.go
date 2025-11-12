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
	Name        string `json:"name"`
	Description string `json:"description"`

	// Parameters is usually satisfied by ToolFunctionParams but left as an
	// any for types from the MCP library to be used directly
	Parameters any `json:"parameters"`
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
	Content          string     `json:"content"`
	Thinking         string     `json:"thinking,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	Images           []string   `json:"images,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// MarshalJSON implements custom JSON marshaling supporting both OpenAI and Anthropic formats
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
	Usage     Usage       `json:"usage"`
}

type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details"`
}

/*
"usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
*/

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

// ============== Message Batches API ==============

// BatchRequest represents a single request in a batch
type BatchRequest struct {
	CustomID string             `json:"custom_id"`
	Params   BatchRequestParams `json:"params"`
}

// BatchRequestParams contains the parameters for a single batch request
type BatchRequestParams struct {
	Model         string    `json:"model"`
	MaxTokens     int       `json:"max_tokens"`
	Messages      []Message `json:"messages"`
	System        string    `json:"system,omitempty"`
	Temperature   float64   `json:"temperature,omitempty"`
	TopP          float64   `json:"top_p,omitempty"`
	TopK          int       `json:"top_k,omitempty"`
	StopSequences []string  `json:"stop_sequences,omitempty"`
	Metadata      any       `json:"metadata,omitempty"`
}

// CreateBatchRequest represents the request to create a new message batch
type CreateBatchRequest struct {
	Requests []BatchRequest `json:"requests"`
}

// BatchRequestCounts tracks the status of requests in a batch
type BatchRequestCounts struct {
	Processing int `json:"processing"`
	Succeeded  int `json:"succeeded"`
	Errored    int `json:"errored"`
	Canceled   int `json:"canceled"`
	Expired    int `json:"expired"`
}

// Batch represents a message batch
type Batch struct {
	ID                string             `json:"id"`
	Type              string             `json:"type"`
	ProcessingStatus  string             `json:"processing_status"`
	RequestCounts     BatchRequestCounts `json:"request_counts"`
	EndedAt           *string            `json:"ended_at,omitempty"`
	CreatedAt         string             `json:"created_at"`
	ExpiresAt         string             `json:"expires_at"`
	ArchivedAt        *string            `json:"archived_at,omitempty"`
	CancelInitiatedAt *string            `json:"cancel_initiated_at,omitempty"`
	ResultsURL        *string            `json:"results_url,omitempty"`
}

// BatchResult represents a single result from a batch
type BatchResult struct {
	CustomID string            `json:"custom_id"`
	Result   BatchResultDetail `json:"result"`
}

// BatchResultDetail contains the result or error for a batch request
type BatchResultDetail struct {
	Type    string              `json:"type"` // "succeeded", "errored", "canceled", "expired"
	Message *BatchMessageResult `json:"message,omitempty"`
	Error   *BatchError         `json:"error,omitempty"`
}

// BatchMessageResult represents the message in a batch result (Anthropic format)
type BatchMessageResult struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Content      []BatchContentBlock `json:"content"`
	Model        string              `json:"model"`
	StopReason   string              `json:"stop_reason,omitempty"`
	StopSequence string              `json:"stop_sequence,omitempty"`
	Usage        BatchUsage          `json:"usage"`
}

// BatchContentBlock represents a content block in the Anthropic response
type BatchContentBlock struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

// BatchUsage represents token usage in batch results
type BatchUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// BatchError represents an error in a batch result
type BatchError struct {
	Type    string           `json:"type"`
	Message string           `json:"message"`
	// Nested error with actual details
	Error   *NestedBatchError `json:"error,omitempty"`
}

// NestedBatchError represents the nested error structure in Anthropic batch errors
type NestedBatchError struct {
	Type    string         `json:"type"`
	Message string         `json:"message"`
	Details map[string]any `json:"details,omitempty"`
}

// GetErrorMessage returns the most descriptive error message available
func (be *BatchError) GetErrorMessage() string {
	if be.Error != nil && be.Error.Message != "" {
		return be.Error.Message
	}
	if be.Message != "" {
		return be.Message
	}
	if be.Error != nil && be.Error.Type != "" {
		return be.Error.Type
	}
	return be.Type
}

// ListBatchesResponse represents the response from listing batches
type ListBatchesResponse struct {
	Data    []Batch `json:"data"`
	HasMore bool    `json:"has_more"`
	FirstID *string `json:"first_id,omitempty"`
	LastID  *string `json:"last_id,omitempty"`
}

// CreateBatch creates a new message batch
func (c *Client) CreateBatch(req CreateBatchRequest) (*Batch, error) {
	resp, err := c.prepareRequest(req, "/messages/batches")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var batch Batch
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, fmt.Errorf("error decoding batch response: %w", err)
	}

	return &batch, nil
}

// GetBatch retrieves the status and details of a specific batch
func (c *Client) GetBatch(batchID string) (*Batch, error) {
	resp, err := c.prepareGet(fmt.Sprintf("/messages/batches/%s", batchID))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var batch Batch
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, fmt.Errorf("error decoding batch response: %w", err)
	}

	return &batch, nil
}

// ListBatches lists all message batches with optional pagination
func (c *Client) ListBatches(limit int, beforeID, afterID string) (*ListBatchesResponse, error) {
	endpoint := "/messages/batches"

	// Build query parameters
	params := []string{}
	if limit > 0 {
		params = append(params, fmt.Sprintf("limit=%d", limit))
	}
	if beforeID != "" {
		params = append(params, fmt.Sprintf("before_id=%s", beforeID))
	}
	if afterID != "" {
		params = append(params, fmt.Sprintf("after_id=%s", afterID))
	}

	if len(params) > 0 {
		endpoint = fmt.Sprintf("%s?%s", endpoint, params[0])
		for i := 1; i < len(params); i++ {
			endpoint = fmt.Sprintf("%s&%s", endpoint, params[i])
		}
	}

	resp, err := c.prepareGet(endpoint)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var listResp ListBatchesResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return nil, fmt.Errorf("error decoding list batches response: %w", err)
	}

	return &listResp, nil
}

// CancelBatch cancels a message batch that is currently processing
func (c *Client) CancelBatch(batchID string) (*Batch, error) {
	resp, err := c.prepareRequest(nil, fmt.Sprintf("/messages/batches/%s/cancel", batchID))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var batch Batch
	if err := json.NewDecoder(resp.Body).Decode(&batch); err != nil {
		return nil, fmt.Errorf("error decoding batch response: %w", err)
	}

	return &batch, nil
}

// GetBatchResults retrieves the results of a completed batch
// Returns a reader for the JSONL format results
func (c *Client) GetBatchResults(batchID string) ([]BatchResult, error) {
	resp, err := c.prepareGet(fmt.Sprintf("/messages/batches/%s/results", batchID))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Read the JSONL response
	var results []BatchResult
	decoder := json.NewDecoder(resp.Body)
	lineNum := 0
	for decoder.More() {
		lineNum++
		var result BatchResult
		if err := decoder.Decode(&result); err != nil {
			return nil, fmt.Errorf("error decoding batch result at line %d: %w", lineNum, err)
		}

		// Debug: print first result and first error to see structure
		if lineNum == 1 {
			debugJSON, _ := json.MarshalIndent(result, "", "  ")
			fmt.Printf("DEBUG: First batch result:\n%s\n", string(debugJSON))
		}
		if result.Result.Type == "errored" && result.Result.Error != nil && lineNum < 100 {
			debugJSON, _ := json.MarshalIndent(result, "", "  ")
			fmt.Printf("DEBUG: Errored result (line %d):\n%s\n", lineNum, string(debugJSON))
		}

		results = append(results, result)
	}

	return results, nil
}
