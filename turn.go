package gollama

import "strings"

// Backend identifies which provider/transport a Client is configured to talk to.
type Backend int

const (
	BackendOpenAI Backend = iota // OpenAI-compatible /chat/completions (also serves Ollama's /v1 endpoint)
	BackendAnthropic
	BackendBedrock
	BackendOllama
)

func (b Backend) String() string {
	switch b {
	case BackendAnthropic:
		return "anthropic"
	case BackendBedrock:
		return "bedrock"
	case BackendOllama:
		return "ollama"
	default:
		return "openai"
	}
}

// Backend reports the provider/transport this client will use. Bedrock and
// Anthropic are detected from explicit configuration (AWS auth / anthropic mode
// or URL). Ollama is heuristically detected from its default port; otherwise the
// client is treated as an OpenAI-compatible endpoint.
//
// Note: Turn routes Ollama through the OpenAI-compatible path, so an Ollama
// endpoint should be configured with a base URL that includes "/v1". The logical
// backend is best tracked by the caller (e.g. ycc's model registry); this accessor
// is a convenience for introspection.
func (c *Client) Backend() Backend {
	switch {
	case c.IsBedrockAPI():
		return BackendBedrock
	case c.IsAnthropicAPI():
		return BackendAnthropic
	case strings.Contains(c.baseURL, ":11434"):
		return BackendOllama
	default:
		return BackendOpenAI
	}
}

// Turn is the canonical, backend-agnostic entry point for a single
// (non-streaming) model turn. It routes to the correct provider based on the
// client's configuration and returns a normalized ResponseMessageGenerate whose
// Choices[0].Message carries the assistant text and any tool calls in a single
// shape regardless of backend.
//
// It delegates to ChatCompletion, which already dispatches Anthropic, Bedrock,
// and OpenAI-compatible endpoints; Turn exists so callers have one stable method
// to depend on and never have to branch per provider. Streaming is always
// disabled (the agent loop consumes whole turns).
func (c *Client) Turn(opts RequestOptions) (*ResponseMessageGenerate, error) {
	opts.Stream = false
	return c.ChatCompletion(opts)
}
