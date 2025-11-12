// Package gollama provides a Go client for interacting with multiple LLM API providers.
// It supports Ollama native API, OpenAI-compatible endpoints, and Anthropic Batch API.
package gollama

import (
	"net/http"
	"time"
)

// Client represents a multi-provider LLM API client.
// It can interact with Ollama, OpenAI-compatible, and Anthropic endpoints
// depending on the baseURL and methods used.
type Client struct {
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
}

// NewClient creates a new LLM API client with the specified base URL.
// The baseURL should point to your API endpoint (e.g., "http://localhost:11434" for Ollama).
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		headers: make(map[string]string),
	}
}

// SetAPIKey sets the API key header for authentication.
// This sets the "x-api-key" header commonly used by cloud LLM providers.
func (c *Client) SetAPIKey(k string) {
	c.SetHeader("x-api-key", k)
}

// SetHeader sets a custom HTTP header for all requests made by this client.
func (c *Client) SetHeader(k, v string) {
	c.headers[k] = v
}
