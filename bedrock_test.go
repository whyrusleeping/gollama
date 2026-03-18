package gollama

import (
	"fmt"
	"os"
	"testing"
)

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
		Model: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
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
		Model:  "global.anthropic.claude-haiku-4-5-20251001-v1:0",
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
		Model: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
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
