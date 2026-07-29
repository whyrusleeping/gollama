package gollama

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// TestBedrockRequestForwardsThinkingAndEffort verifies that the Bedrock
// InvokeModel body carries the same reasoning controls as the Anthropic direct
// body. Before these fields existed on bedrockRequest, RequestOptions.Thinking
// and .Effort were silently dropped on the Bedrock path.
func TestBedrockRequestForwardsThinkingAndEffort(t *testing.T) {
	req, err := buildBedrockRequest(RequestOptions{
		Model:           BedrockOpus5,
		Messages:        []Message{{Role: "user", Content: "hi"}},
		Thinking:        "adaptive",
		ThinkingDisplay: "summarized",
		Effort:          "high",
	})
	if err != nil {
		t.Fatal(err)
	}
	if req.Thinking == nil || req.Thinking.Type != "adaptive" || req.Thinking.Display != "summarized" {
		t.Fatalf("thinking = %+v, want adaptive/summarized", req.Thinking)
	}
	if req.OutputConfig == nil || req.OutputConfig.Effort != "high" {
		t.Fatalf("output_config = %+v, want effort=high", req.OutputConfig)
	}

	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	body := string(b)
	// budget_tokens is not modeled at all; current models reject it.
	if strings.Contains(body, "budget_tokens") {
		t.Fatalf("request must not contain budget_tokens: %s", body)
	}
	if !strings.Contains(body, `"thinking":{"type":"adaptive","display":"summarized"}`) {
		t.Fatalf("serialized request missing thinking: %s", body)
	}
	if !strings.Contains(body, `"output_config":{"effort":"high"}`) {
		t.Fatalf("serialized request missing output_config: %s", body)
	}
	// The Bedrock body must never carry a model field; the model is in the URL.
	if strings.Contains(body, `"model"`) {
		t.Fatalf("bedrock request must not contain a model field: %s", body)
	}

	// Unset by default: no thinking or output_config keys at all.
	req2, err := buildBedrockRequest(RequestOptions{
		Model:    BedrockOpus5,
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if req2.Thinking != nil || req2.OutputConfig != nil {
		t.Fatalf("expected no thinking/output_config by default; got %+v / %+v", req2.Thinking, req2.OutputConfig)
	}
	b2, err := json.Marshal(req2)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b2), "thinking") || strings.Contains(string(b2), "output_config") {
		t.Fatalf("default bedrock request should omit reasoning keys: %s", b2)
	}
}

// TestBedrockRequestReplaysThinkingBlocks verifies that thinking blocks captured
// from a previous assistant turn ride along inside messages on the Bedrock path
// too, since it delegates to the shared Anthropic request builder.
func TestBedrockRequestReplaysThinkingBlocks(t *testing.T) {
	req, err := buildBedrockRequest(RequestOptions{
		Model: BedrockOpus5,
		Messages: []Message{
			{Role: "user", Content: "what is the weather"},
			{
				Role: "assistant",
				ThinkingBlocks: []ThinkingBlock{
					{Thinking: "let me check", Signature: "sig-abc"},
					{Redacted: "opaque-payload"},
				},
				ToolCalls: []ToolCall{{
					ID:       "call_1",
					Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"SF"}`},
				}},
			},
			{Role: "tool", ToolCallID: "call_1", Content: "sunny"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	body := string(b)
	for _, want := range []string{`"signature":"sig-abc"`, `"redacted_thinking"`, `"data":"opaque-payload"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("bedrock body missing %s: %s", want, body)
		}
	}
	// Thinking must precede tool_use in the assistant content array.
	if strings.Index(body, `"thinking"`) > strings.Index(body, `"tool_use"`) {
		t.Fatalf("thinking blocks must precede tool_use: %s", body)
	}
}

func getBedrockClient(t *testing.T) *Client {
	accessKey := os.Getenv("AWS_ACCESS_KEY_ID")
	secretKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = "us-east-1"
	}

	if accessKey == "" || secretKey == "" {
		t.Skip("AWS credentials not available (set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)")
	}

	sessionToken := os.Getenv("AWS_SESSION_TOKEN")
	return NewBedrockClient(region, accessKey, secretKey, sessionToken)
}

func TestBedrockChatCompletion(t *testing.T) {
	client := getBedrockClient(t)

	resp, err := client.ChatCompletion(RequestOptions{
		Model: BedrockHaiku45,
		Messages: []Message{
			{Role: "user", Content: "Say hello in exactly three words."},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	fmt.Printf("Model: %s\n", resp.Model)
	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Printf("Usage: input=%d output=%d total=%d\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)

	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("unexpected role: %s", resp.Choices[0].Message.Role)
	}
	if resp.Choices[0].Message.Content == "" {
		t.Error("empty response content")
	}
	if resp.Usage.PromptTokens == 0 {
		t.Error("expected non-zero prompt tokens")
	}
	if resp.Usage.CompletionTokens == 0 {
		t.Error("expected non-zero completion tokens")
	}
}

func TestBedrockWithSystemPrompt(t *testing.T) {
	client := getBedrockClient(t)

	resp, err := client.ChatCompletion(RequestOptions{
		Model:  BedrockHaiku45,
		System: "You are a pirate. Respond in pirate speak.",
		Messages: []Message{
			{Role: "user", Content: "What is 2+2?"},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
}

func TestBedrockWithToolUse(t *testing.T) {
	client := getBedrockClient(t)

	resp, err := client.ChatCompletion(RequestOptions{
		Model: BedrockHaiku45,
		Messages: []Message{
			{Role: "user", Content: "What's the weather in Seattle?"},
		},
		Tools: []ToolParam{
			{
				Type: "function",
				Function: &ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather for a location",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "City name",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) == 0 {
		t.Fatalf("Expected tool call, got text: %s", msg.Content)
	}

	tc := msg.ToolCalls[0]
	fmt.Printf("Tool call: %s(%s)\n", tc.Function.Name, tc.Function.Arguments)

	if tc.Function.Name != "get_weather" {
		t.Errorf("unexpected tool name: %s", tc.Function.Name)
	}
}

func TestBedrockCaching(t *testing.T) {
	client := getBedrockClient(t)

	// System prompt needs to be at least 1024 tokens for caching to kick in.
	// Pad it well past that threshold.
	systemPrompt := "You are a helpful assistant that answers questions concisely.\n\n" +
		strings.Repeat("Remember: always be concise, accurate, and helpful in your responses. ", 200)

	messages := []Message{
		{Role: "user", Content: "What is the capital of France?"},
	}

	// First request — should create the cache entry
	fmt.Println("=== Request 1: expecting cache creation ===")
	resp1, err := client.ChatCompletion(RequestOptions{
		Model:    BedrockSonnet46,
		System:   systemPrompt,
		Messages: messages,
	})
	if err != nil {
		t.Fatalf("First request failed: %v", err)
	}

	fmt.Printf("Response: %s\n", resp1.Choices[0].Message.Content)
	fmt.Printf("Usage: input=%d output=%d cache_create=%d cache_read=%d\n",
		resp1.Usage.PromptTokens, resp1.Usage.CompletionTokens,
		resp1.Usage.CacheCreationInputTokens, resp1.Usage.CacheReadInputTokens)

	if resp1.Usage.CacheCreationInputTokens == 0 {
		t.Error("expected cache creation tokens on first request")
	}

	// Add the response to conversation history
	messages = append(messages, resp1.Choices[0].Message)
	messages = append(messages, Message{
		Role:    "user",
		Content: "What about Germany?",
	})

	// Second request — same system prompt prefix, should get cache hits
	fmt.Println("\n=== Request 2: expecting cache read ===")
	resp2, err := client.ChatCompletion(RequestOptions{
		Model:    BedrockSonnet46,
		System:   systemPrompt,
		Messages: messages,
	})
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}

	fmt.Printf("Response: %s\n", resp2.Choices[0].Message.Content)
	fmt.Printf("Usage: input=%d output=%d cache_create=%d cache_read=%d\n",
		resp2.Usage.PromptTokens, resp2.Usage.CompletionTokens,
		resp2.Usage.CacheCreationInputTokens, resp2.Usage.CacheReadInputTokens)

	if resp2.Usage.CacheReadInputTokens == 0 {
		t.Error("expected cache read tokens on second request")
	}

	fmt.Printf("\n=== Summary ===\n")
	fmt.Printf("Cached tokens read on request 2: %d\n", resp2.Usage.CacheReadInputTokens)
}
