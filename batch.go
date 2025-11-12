package gollama

import (
	"encoding/json"
	"fmt"
)

// ============== Message Batches API ==============
// Anthropic Batch API types and methods

// BatchRequest represents a single request in a batch.
type BatchRequest struct {
	CustomID string             `json:"custom_id"`
	Params   BatchRequestParams `json:"params"`
}

// BatchRequestParams contains the parameters for a single batch request.
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

// CreateBatchRequest represents the request to create a new message batch.
type CreateBatchRequest struct {
	Requests []BatchRequest `json:"requests"`
}

// BatchRequestCounts tracks the status of requests in a batch.
type BatchRequestCounts struct {
	Processing int `json:"processing"`
	Succeeded  int `json:"succeeded"`
	Errored    int `json:"errored"`
	Canceled   int `json:"canceled"`
	Expired    int `json:"expired"`
}

// Batch represents a message batch with status and metadata.
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

// BatchResult represents a single result from a batch.
type BatchResult struct {
	CustomID string            `json:"custom_id"`
	Result   BatchResultDetail `json:"result"`
}

// BatchResultDetail contains the result or error for a batch request.
type BatchResultDetail struct {
	Type    string              `json:"type"` // "succeeded", "errored", "canceled", "expired"
	Message *BatchMessageResult `json:"message,omitempty"`
	Error   *BatchError         `json:"error,omitempty"`
}

// BatchMessageResult represents the message in a batch result (Anthropic format).
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

// BatchContentBlock represents a content block in the Anthropic response.
type BatchContentBlock struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

// BatchUsage represents token usage in batch results.
type BatchUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// BatchError represents an error in a batch result.
type BatchError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
	// Nested error with actual details
	Error *NestedBatchError `json:"error,omitempty"`
}

// NestedBatchError represents the nested error structure in Anthropic batch errors.
type NestedBatchError struct {
	Type    string         `json:"type"`
	Message string         `json:"message"`
	Details map[string]any `json:"details,omitempty"`
}

// GetErrorMessage returns the most descriptive error message available.
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

// ListBatchesResponse represents the response from listing batches.
type ListBatchesResponse struct {
	Data    []Batch `json:"data"`
	HasMore bool    `json:"has_more"`
	FirstID *string `json:"first_id,omitempty"`
	LastID  *string `json:"last_id,omitempty"`
}

// CreateBatch creates a new message batch.
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

// GetBatch retrieves the status and details of a specific batch.
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

// ListBatches lists all message batches with optional pagination.
// Set limit to 0 to use the API default. Use beforeID or afterID for pagination.
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

// CancelBatch cancels a message batch that is currently processing.
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

// GetBatchResults retrieves the results of a completed batch.
// Returns a slice of BatchResult parsed from the JSONL format response.
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

		results = append(results, result)
	}

	return results, nil
}
