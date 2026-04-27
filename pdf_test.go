package gollama

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// secretPhrase is the unique sentence inside testdata/secret.pdf. The test
// verifies the model can read it back through a tool result document block,
// so it has to be specific enough that the model couldn't guess it from the
// prompt alone. Keep in sync with testdata/genpdf.go.
const secretPhrase = "XYZZY42-NOVEMBER-CHESHIRE"

// TestAnthropicToolResultDocument round-trips a PDF through a tool result.
//
// Flow:
//  1. The model is asked to call get_document.
//  2. We return the PDF in ToolResult.Documents.
//  3. The model reads the PDF and the response should mention the secret.
//
// This requires ANTHROPIC_API_KEY; skipped otherwise so `go test ./...` is
// safe in CI without credentials.
func TestAnthropicToolResultDocument(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set; skipping live API test")
	}

	pdfBytes, err := os.ReadFile("testdata/secret.pdf")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	pdfB64 := base64.StdEncoding.EncodeToString(pdfBytes)

	client := NewClient("https://api.anthropic.com")
	client.SetAPIKey(apiKey)
	client.SetHeader("anthropic-version", "2023-06-01")

	tools := []ToolParam{
		{
			Type: "function",
			Function: &ToolFunction{
				Name:        "get_document",
				Description: "Returns the secret document as a PDF. Call this when the user asks about the document.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
					"required":   []string{},
				},
			},
		},
	}

	model := "claude-sonnet-4-20250514"

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant. When the user asks about the document, call get_document and then answer based on its contents."},
		{Role: "user", Content: "Use the get_document tool to fetch the document, then tell me exactly what secret password it contains."},
	}

	// First request: expect a tool call.
	resp1, err := client.ChatCompletion(RequestOptions{
		Model:    model,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("first request: %v", err)
	}
	if len(resp1.Choices) == 0 {
		t.Fatal("no choices in first response")
	}

	asstMsg := resp1.Choices[0].Message
	if len(asstMsg.ToolCalls) == 0 {
		t.Fatalf("expected a tool call, got text: %q", asstMsg.Content)
	}

	var targetCall *ToolCall
	for i := range asstMsg.ToolCalls {
		if asstMsg.ToolCalls[i].Function.Name == "get_document" {
			targetCall = &asstMsg.ToolCalls[i]
			break
		}
	}
	if targetCall == nil {
		t.Fatalf("model called something other than get_document: %+v", asstMsg.ToolCalls)
	}

	// Echo back the assistant turn, then the tool result with the PDF document.
	messages = append(messages, asstMsg)
	messages = append(messages, Message{
		Role:       "tool",
		ToolCallID: targetCall.ID,
		Content:    "Document attached.",
		Documents: []Document{
			{
				Base64:    pdfB64,
				MediaType: "application/pdf",
				Title:     "secret.pdf",
			},
		},
	})

	// Second request: model should read the PDF and surface the secret.
	resp2, err := client.ChatCompletion(RequestOptions{
		Model:    model,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("second request: %v", err)
	}
	if len(resp2.Choices) == 0 {
		t.Fatal("no choices in second response")
	}

	final := resp2.Choices[0].Message
	out := final.Content
	if out == "" {
		// Some models emit further tool calls instead of text — surface that for debugging.
		if len(final.ToolCalls) > 0 {
			b, _ := json.Marshal(final.ToolCalls)
			t.Fatalf("expected text response, got tool calls: %s", string(b))
		}
		t.Fatal("empty response content")
	}

	t.Logf("model response: %s", out)

	if !strings.Contains(strings.ToUpper(out), secretPhrase) {
		t.Fatalf("response did not include secret phrase %q.\nfull response: %s", secretPhrase, out)
	}

	fmt.Printf("usage: input=%d output=%d cache_read=%d cache_write=%d\n",
		resp2.Usage.PromptTokens, resp2.Usage.CompletionTokens,
		resp2.Usage.CacheReadInputTokens, resp2.Usage.CacheCreationInputTokens)
}
