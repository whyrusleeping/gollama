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

// isRetryableStatus returns true for status codes that should trigger a retry.
func isRetryableStatus(code int) bool {
	return code == 429 || code == 529 || code == 503
}

// prepareRequest creates and sends a POST request to the specified endpoint.
// It marshals the body, sets headers, and validates the response status.
// Retries with exponential backoff on 429, 503, and 529 errors.
func (c *Client) prepareRequest(body any, endpoint string) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	maxRetries := 5
	baseDelay := 5 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
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

		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		// Check if we should retry
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

// prepareGet creates and sends a GET request to the specified endpoint.
// It sets headers and validates the response status.
// Retries with exponential backoff on 429, 503, and 529 errors.
func (c *Client) prepareGet(endpoint string) (*http.Response, error) {
	maxRetries := 5
	baseDelay := 5 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
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

		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		// Check if we should retry
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
