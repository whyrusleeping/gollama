package gollama

import "testing"

// TestOpenAIOmitsNestedOptions verifies that generation parameters supplied via
// RequestOptions.Options are promoted to top-level fields (max_tokens, ...) and
// that the nested Ollama-style "options" object is NOT serialized for the
// OpenAI-compatible backend. Strict servers (e.g. Fireworks) reject requests
// carrying unknown fields, so the nested object must ride to Ollama only.
func TestOpenAIOmitsNestedOptions(t *testing.T) {
	var got map[string]any
	srv := captureServer(t, "", &got)
	defer srv.Close()
	c := openaiTestClient(t, srv.URL)

	if _, err := c.Turn(RequestOptions{
		Model:    "gpt-5.1",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Options:  &Options{MaxTokens: 123},
	}); err != nil {
		t.Fatalf("Turn: %v", err)
	}

	if v, ok := got["options"]; ok {
		t.Errorf("nested options object must be absent on OpenAI backend, got %v", v)
	}
	if mt, ok := got["max_tokens"].(float64); !ok || int(mt) != 123 {
		t.Errorf("max_tokens = %v, want 123 (promoted from Options)", got["max_tokens"])
	}
}

// TestOllamaKeepsNestedOptions verifies the Ollama backend still receives the
// nested options object it expects.
func TestOllamaKeepsNestedOptions(t *testing.T) {
	var got map[string]any
	srv := captureServer(t, "", &got)
	defer srv.Close()
	c := ollamaBackedClient(t, srv.URL)

	if _, err := c.Turn(RequestOptions{
		Model:    "llama3",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Options:  &Options{MaxTokens: 123},
	}); err != nil {
		t.Fatalf("Turn: %v", err)
	}

	opts, ok := got["options"].(map[string]any)
	if !ok {
		t.Fatalf("nested options object missing on Ollama backend; body=%v", got)
	}
	if mt, ok := opts["num_predict"].(float64); !ok || int(mt) != 123 {
		t.Errorf("options.num_predict = %v, want 123", opts["num_predict"])
	}
}
