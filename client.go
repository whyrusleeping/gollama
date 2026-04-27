// Package gollama provides a Go client for interacting with multiple LLM API providers.
// It supports Ollama native API, OpenAI-compatible endpoints, and Anthropic Batch API.
package gollama

import (
	"net/http"
	"strings"
	"time"
)

// Client represents a multi-provider LLM API client.
// It can interact with Ollama, OpenAI-compatible, Anthropic, and AWS Bedrock endpoints
// depending on the baseURL and methods used.
type Client struct {
	baseURL       string
	httpClient    *http.Client
	headers       map[string]string
	bedrock       *BedrockConfig
	anthropicMode *bool // nil = auto-detect from URL; non-nil = explicit override
}

// NewClient creates a new LLM API client with the specified base URL.
// The baseURL should point to your API endpoint (e.g., "http://localhost:11434" for Ollama).
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 300 * time.Second,
		},
		headers: make(map[string]string),
	}
}

// SetAnthropicMode explicitly enables or disables Anthropic native API mode.
// By default, the client auto-detects Anthropic endpoints from the base URL.
// Use this when routing through a proxy or gateway on a custom domain.
func (c *Client) SetAnthropicMode(enabled bool) {
	c.anthropicMode = &enabled
}

// SetAPIKey sets the "x-api-key" header used by Anthropic's API.
// For OpenAI-compatible APIs that use Bearer tokens, use SetBearerToken instead.
func (c *Client) SetAPIKey(k string) {
	c.SetHeader("x-api-key", k)
}

// SetBearerToken sets the "Authorization: Bearer" header used by OpenAI-compatible APIs.
func (c *Client) SetBearerToken(k string) {
	c.SetHeader("Authorization", "Bearer "+k)
}

// SetHeader sets a custom HTTP header for all requests made by this client.
func (c *Client) SetHeader(k, v string) {
	c.headers[k] = v
}

// anthropicEndpoint returns the correct API path for Anthropic endpoints,
// accounting for whether the baseURL already includes the /v1 prefix.
func (c *Client) anthropicEndpoint(path string) string {
	if strings.HasSuffix(c.baseURL, "/v1") {
		return path
	}
	return "/v1" + path
}
