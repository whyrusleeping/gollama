package gollama

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestAnthropicCachingWithToolCalls(t *testing.T) {
	// Read API key from file or environment
	apiKey := os.Getenv("ANTHROPIC_API_KEY")

	client := NewClient("https://api.anthropic.com")
	client.SetAPIKey(apiKey)
	client.SetHeader("anthropic-version", "2023-06-01")

	// Define a tool that generates a lot of output (simulating code generation)
	tools := []ToolParam{
		{
			Type: "function",
			Function: &ToolFunction{
				Name:        "generate_code",
				Description: "Generates code based on the given specification. Always use this tool when asked to write code.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"code": map[string]any{
							"type":        "string",
							"description": "The generated code",
						},
						"language": map[string]any{
							"type":        "string",
							"description": "The programming language",
						},
					},
					"required": []string{"code", "language"},
				},
			},
		},
	}

	// Large system prompt to make caching effects more visible
	systemPrompt := `You are a helpful coding assistant. When asked to write code, you MUST use the generate_code tool to provide the code. Never write code directly in your response - always use the tool.

Here are some guidelines for code generation:
` + strings.Repeat("- Write clean, well-documented code\n", 100) // Pad to make it larger

	// First request: ask for code generation (should trigger tool call)
	messages := []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: "Please write a Python function that implements a binary search tree with insert, delete, and search operations. Include comprehensive docstrings and type hints. Make it at least 200 lines of code."},
	}

	fmt.Println("=== Request 1: Initial request (expecting tool call) ===")
	resp1, err := client.ChatCompletion(RequestOptions{
		Model:    "claude-sonnet-4-20250514",
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("First request failed: %v", err)
	}

	fmt.Printf("Response 1 Usage:\n")
	fmt.Printf("  Input tokens: %d\n", resp1.Usage.PromptTokens)
	fmt.Printf("  Output tokens: %d\n", resp1.Usage.CompletionTokens)
	fmt.Printf("  Cache read: %d\n", resp1.Usage.CacheReadInputTokens)
	fmt.Printf("  Cache write: %d\n", resp1.Usage.CacheCreationInputTokens)

	if len(resp1.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	msg1 := resp1.Choices[0].Message
	if len(msg1.ToolCalls) == 0 {
		t.Fatalf("Expected tool call, got text response: %s", msg1.Content)
	}

	fmt.Printf("Tool call: %s\n", msg1.ToolCalls[0].Function.Name)
	fmt.Printf("Tool call arguments length: %d chars\n", len(msg1.ToolCalls[0].Function.Arguments))

	// Add assistant message with tool call to history
	messages = append(messages, msg1)

	// Add tool result (small response)
	toolResult := Message{
		Role:       "tool",
		ToolCallID: msg1.ToolCalls[0].ID,
		Content:    "Code saved successfully to bst.py",
	}
	messages = append(messages, toolResult)

	// Second request: send tool result back
	fmt.Println("\n=== Request 2: After tool result (assistant message should be cached) ===")
	resp2, err := client.ChatCompletion(RequestOptions{
		Model:    "claude-sonnet-4-20250514",
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}

	fmt.Printf("Response 2 Usage:\n")
	fmt.Printf("  Input tokens: %d\n", resp2.Usage.PromptTokens)
	fmt.Printf("  Output tokens: %d\n", resp2.Usage.CompletionTokens)
	fmt.Printf("  Cache read: %d\n", resp2.Usage.CacheReadInputTokens)
	fmt.Printf("  Cache write: %d\n", resp2.Usage.CacheCreationInputTokens)

	// Add this response to history
	if len(resp2.Choices) > 0 {
		messages = append(messages, resp2.Choices[0].Message)
	}

	// Third request: ask a follow-up question (should hit cache for previous messages)
	messages = append(messages, Message{
		Role:    "user",
		Content: "Can you add a method to find the minimum value in the tree?",
	})

	fmt.Println("\n=== Request 3: Follow-up question (should see cache hits) ===")
	resp3, err := client.ChatCompletion(RequestOptions{
		Model:    "claude-sonnet-4-20250514",
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		t.Fatalf("Third request failed: %v", err)
	}

	fmt.Printf("Response 3 Usage:\n")
	fmt.Printf("  Input tokens: %d\n", resp3.Usage.PromptTokens)
	fmt.Printf("  Output tokens: %d\n", resp3.Usage.CompletionTokens)
	fmt.Printf("  Cache read: %d\n", resp3.Usage.CacheReadInputTokens)
	fmt.Printf("  Cache write: %d\n", resp3.Usage.CacheCreationInputTokens)

	// Analysis
	fmt.Println("\n=== Analysis ===")
	if resp2.Usage.CacheReadInputTokens > 0 {
		fmt.Println("✓ Request 2 had cache hits - system prompt and/or tools were cached")
	} else {
		fmt.Println("✗ Request 2 had no cache hits")
	}

	if resp3.Usage.CacheReadInputTokens > resp2.Usage.CacheReadInputTokens {
		fmt.Println("✓ Request 3 had more cache hits than Request 2 - previous messages are being cached")
	} else if resp3.Usage.CacheReadInputTokens > 0 {
		fmt.Println("~ Request 3 had cache hits but not more than Request 2")
	} else {
		fmt.Println("✗ Request 3 had no cache hits")
	}

	// Check if the tool call content (which could be large) is contributing to cache
	toolCallSize := len(msg1.ToolCalls[0].Function.Arguments)
	fmt.Printf("\nTool call argument size: %d chars (~%d tokens)\n", toolCallSize, toolCallSize/4)
	fmt.Printf("Request 3 cache read: %d tokens\n", resp3.Usage.CacheReadInputTokens)

	if resp3.Usage.CacheReadInputTokens > toolCallSize/4 {
		fmt.Println("✓ Cache read tokens exceed tool call size - tool_use content is likely cached")
	} else {
		fmt.Println("? Cache read tokens less than tool call size - tool_use content may not be fully cached")
	}
}
