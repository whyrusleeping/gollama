package gollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

const (
	maxRetries = 5
	baseDelay  = 5 * time.Second
)

// isRetryableStatus returns true for status codes that should trigger a retry.
func isRetryableStatus(code int) bool {
	return code == 429 || code == 529 || code == 503
}

// doWithRetry executes an HTTP request with exponential backoff on retryable errors.
// The newReq function is called on each attempt to produce a fresh *http.Request
// (necessary for POST bodies, which are consumed on each attempt).
func (c *Client) doWithRetry(newReq func() (*http.Request, error)) (*http.Response, error) {
	for attempt := 0; attempt <= maxRetries; attempt++ {
		req, err := newReq()
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

		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if isRetryableStatus(resp.StatusCode) && attempt < maxRetries {
			delay := baseDelay * time.Duration(1<<attempt) // exponential: 5s, 10s, 20s, 40s, 80s
			log.Printf("API returned %d, retrying in %v (attempt %d/%d)", resp.StatusCode, delay, attempt+1, maxRetries)
			time.Sleep(delay)
			continue
		}

		return nil, fmt.Errorf("API returned non-200 status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	return nil, fmt.Errorf("max retries exceeded")
}

// prepareRequest creates and sends a POST request to the specified endpoint.
// It marshals the body, sets headers, and validates the response status.
// Retries with exponential backoff on 429, 503, and 529 errors.
func (c *Client) prepareRequest(body any, endpoint string) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	url := c.baseURL + endpoint
	return c.doWithRetry(func() (*http.Request, error) {
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(data))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		return req, nil
	})
}

// prepareGet creates and sends a GET request to the specified endpoint.
// It sets headers and validates the response status.
// Retries with exponential backoff on 429, 503, and 529 errors.
func (c *Client) prepareGet(endpoint string) (*http.Response, error) {
	url := c.baseURL + endpoint
	return c.doWithRetry(func() (*http.Request, error) {
		return http.NewRequest("GET", url, nil)
	})
}
