package gollama

import (
	"encoding/json"
	"fmt"
)

// ListModels retrieves the list of available models from the OpenAI-compatible /models endpoint.
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

// ChatCompletion sends a chat completion request to the OpenAI-compatible /chat/completions endpoint.
// Returns a ResponseMessageGenerate with choices and usage information.
// TODO: Streaming not yet supported
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
