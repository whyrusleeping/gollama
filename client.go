// Package gollama provides a Go client for interacting with multiple LLM API providers.
// It supports Ollama native API, OpenAI-compatible endpoints, and Anthropic Batch API.
package gollama

import (
	"net/http"
	"strings"
	"time"
)

// defaultHTTPTimeout is the total-request timeout of the default HTTP client.
// Note that it includes the time spent reading a streamed response body.
const defaultHTTPTimeout = 300 * time.Second

// Client represents a multi-provider LLM API client.
// It can interact with Ollama, OpenAI-compatible, Anthropic, and AWS Bedrock endpoints
// depending on the baseURL and methods used.
type Client struct {
	baseURL       string
	httpClient    *http.Client
	headers       map[string]string
	bedrock       *BedrockConfig
	anthropicMode *bool // nil = auto-detect from URL; non-nil = explicit override
	maxRetries    *int  // nil = default (defaultMaxRetries); non-nil = explicit override
}

// NewClient creates a new LLM API client with the specified base URL.
// The baseURL should point to your API endpoint (e.g., "http://localhost:11434" for Ollama).
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: defaultHTTPTimeout,
		},
		headers: make(map[string]string),
	}
}

// SetHTTPClient replaces the HTTP client used for all requests. Use it to
// control transport policy — for example to drop the default 300-second total
// timeout (which also bounds streamed response bodies) in favour of caller
// context deadlines and a stream-inactivity watchdog. A nil client restores the
// default. The client is used as-is; configured request headers are still
// applied per request.
func (c *Client) SetHTTPClient(hc *http.Client) {
	if hc == nil {
		hc = &http.Client{Timeout: defaultHTTPTimeout}
	}
	c.httpClient = hc
}

// SetAnthropicMode explicitly enables or disables Anthropic native API mode.
// By default, the client auto-detects Anthropic endpoints from the base URL.
// Use this when routing through a proxy or gateway on a custom domain.
func (c *Client) SetAnthropicMode(enabled bool) {
	c.anthropicMode = &enabled
}

// SetMaxRetries configures how many times the HTTP transport retries a request
// after retryable responses (HTTP 429, 503, 529) before giving up. n is the
// number of retries after the initial attempt, so n=0 disables transport-level
// retry entirely (the request is attempted exactly once). Negative values are
// clamped to 0. When unset, the default of defaultMaxRetries retries applies.
func (c *Client) SetMaxRetries(n int) {
	if n < 0 {
		n = 0
	}
	c.maxRetries = &n
}

// effectiveMaxRetries returns the configured transport retry count, or the
// package default when SetMaxRetries has not been called.
func (c *Client) effectiveMaxRetries() int {
	if c.maxRetries != nil {
		return *c.maxRetries
	}
	return defaultMaxRetries
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
