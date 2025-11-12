package gollama

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// Generate sends a completion request to the Ollama /api/generate endpoint.
// Returns a GenerateResponse with the generated text and metadata.
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

// handleGenerateResponse processes a non-streaming generate response.
func (c *Client) handleGenerateResponse(resp *http.Response) (*GenerateResponse, error) {
	decoder := json.NewDecoder(resp.Body)
	var response GenerateResponse
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &response, nil
}

// handleGenerateStream processes a streaming generate response.
// TODO: Implement full streaming support - currently returns first chunk only
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

// Chat sends a chat completion request to the Ollama /api/chat endpoint.
// Returns a ResponseMessage with the assistant's reply and metadata.
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

// handleChatResponse processes a non-streaming chat response.
func (c *Client) handleChatResponse(resp *http.Response) (*ResponseMessage, error) {
	decoder := json.NewDecoder(resp.Body)
	var response ResponseMessage
	if err := decoder.Decode(&response); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &response, nil
}

// handleChatStream processes a streaming chat response.
// TODO: Implement full streaming support
func (c *Client) handleChatStream(resp *http.Response) (*ResponseMessage, error) {
	return nil, fmt.Errorf("not dealing with streams yet")
}
