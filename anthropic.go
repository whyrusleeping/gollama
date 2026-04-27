package gollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Anthropic native API types

type anthropicRequest struct {
	Model     string                 `json:"model"`
	MaxTokens int                    `json:"max_tokens"`
	System    []anthropicSystemBlock `json:"system,omitempty"`
	Messages  []anthropicMessage     `json:"messages"`
	Tools     []anthropicTool        `json:"tools,omitempty"`
}

type anthropicSystemBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicCacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

type anthropicMessage struct {
	Role    string        `json:"role"`
	Content []interface{} `json:"content"`
}

type anthropicTextBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicToolUseBlock struct {
	Type         string                 `json:"type"`
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Input        any                    `json:"input"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicToolResultBlock struct {
	Type         string                 `json:"type"`
	ToolUseID    string                 `json:"tool_use_id"`
	Content      []interface{}          `json:"content"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicImageBlock struct {
	Type   string               `json:"type"`
	Source anthropicImageSource `json:"source"`
}

type anthropicImageSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // required for base64
	Data      string `json:"data,omitempty"`       // base64 data
	URL       string `json:"url,omitempty"`         // URL source
}

type anthropicDocumentBlock struct {
	Type   string                  `json:"type"` // "document"
	Source anthropicDocumentSource `json:"source"`
	Title  string                  `json:"title,omitempty"`
}

type anthropicDocumentSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // required for base64 (e.g. "application/pdf")
	Data      string `json:"data,omitempty"`       // base64 data
	URL       string `json:"url,omitempty"`         // URL source
}

// buildAnthropicDocumentBlock constructs a document block from a Document,
// preferring URL source when set, else base64.
func buildAnthropicDocumentBlock(doc Document) anthropicDocumentBlock {
	block := anthropicDocumentBlock{
		Type:  "document",
		Title: doc.Title,
	}
	if doc.URL != "" {
		block.Source = anthropicDocumentSource{
			Type: "url",
			URL:  doc.URL,
		}
	} else {
		mediaType := doc.MediaType
		if mediaType == "" {
			mediaType = "application/pdf"
		}
		block.Source = anthropicDocumentSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      doc.Base64,
		}
	}
	return block
}

type anthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  any                    `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []anthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

type anthropicContentBlock struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// buildAnthropicRequest converts generic RequestOptions into an Anthropic-native request struct.
// This is shared by both the direct Anthropic API and the Bedrock API paths.
func buildAnthropicRequest(opts RequestOptions) (*anthropicRequest, error) {
	req := &anthropicRequest{
		Model:     opts.Model,
		MaxTokens: 8192, // Default max tokens
	}

	if opts.Options != nil && opts.Options.MaxTokens > 0 {
		req.MaxTokens = opts.Options.MaxTokens
	}

	// Convert system prompt to Anthropic format with caching.
	// Priority: SystemBlocks > System string > messages with role "system"
	// Any role="system" message in opts.Messages is consumed here (regardless
	// of position) since the native /v1/messages API rejects it as a chat role.
	if len(opts.SystemBlocks) > 0 {
		for _, block := range opts.SystemBlocks {
			sb := anthropicSystemBlock{
				Type: "text",
				Text: block.Text,
			}
			if block.Cache {
				sb.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
			}
			req.System = append(req.System, sb)
		}
	} else if opts.System != "" {
		req.System = []anthropicSystemBlock{
			{
				Type:         "text",
				Text:         opts.System,
				CacheControl: &anthropicCacheControl{Type: "ephemeral"},
			},
		}
	} else {
		// Promote any role="system" messages to req.System (joined with blank lines).
		var sysParts []string
		for _, m := range opts.Messages {
			if m.Role == "system" && m.Content != "" {
				sysParts = append(sysParts, m.Content)
			}
		}
		if len(sysParts) > 0 {
			req.System = []anthropicSystemBlock{
				{
					Type:         "text",
					Text:         strings.Join(sysParts, "\n\n"),
					CacheControl: &anthropicCacheControl{Type: "ephemeral"},
				},
			}
		}
	}

	// Convert tools, optionally caching the last tool.
	// When SystemBlocks is provided, the caller manages cache breakpoints explicitly,
	// so we skip the tools breakpoint to stay within Anthropic's 4-breakpoint limit.
	cacheTools := len(opts.SystemBlocks) == 0
	if len(opts.Tools) > 0 {
		for i, t := range opts.Tools {
			tool := anthropicTool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: t.Function.Parameters,
			}
			if cacheTools && i == len(opts.Tools)-1 {
				tool.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
			}
			req.Tools = append(req.Tools, tool)
		}
	}

	// Convert messages to Anthropic format
	// Strategy: Anthropic allows max 4 cache breakpoints. We use them as:
	// 1. System prompt (cached above)
	// 2. Last tool definition (cached above)
	// 3. Last assistant message (often contains large tool_use blocks with generated code)
	// 4. Last user/tool_result message
	// This ensures that when a tool returns a small result after a large tool_use,
	// the assistant's tool_use content is cached and doesn't need to be reprocessed.

	// Find the last assistant message index for caching (skipping system msgs)
	lastAssistantIdx := -1
	for i := 0; i < len(opts.Messages); i++ {
		if opts.Messages[i].Role == "assistant" {
			lastAssistantIdx = i
		}
	}

	for i := 0; i < len(opts.Messages); i++ {
		msg := opts.Messages[i]

		// System messages were promoted to req.System above; skip them here.
		if msg.Role == "system" {
			continue
		}

		antMsg := anthropicMessage{
			Role:    msg.Role,
			Content: []any{},
		}

		if msg.Role == "tool" {
			// Tool result - build content as array of blocks
			antMsg.Role = "user"

			var resultContent []any

			// Add text content
			if msg.Content != "" {
				resultContent = append(resultContent, anthropicTextBlock{
					Type: "text",
					Text: msg.Content,
				})
			}

			// Add images if present
			for _, img := range msg.Images {
				resultContent = append(resultContent, anthropicImageBlock{
					Type: "image",
					Source: anthropicImageSource{
						Type:      "base64",
						MediaType: DetectImageMediaType(img),
						Data:      img,
					},
				})
			}

			// Add documents if present (e.g. PDFs)
			for _, doc := range msg.Documents {
				resultContent = append(resultContent, buildAnthropicDocumentBlock(doc))
			}

			toolResult := anthropicToolResultBlock{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Content:   resultContent,
			}

			// consecutive tool results need to be in a single message, so merge them here
			if len(req.Messages) > 0 {
				lastMsg := &req.Messages[len(req.Messages)-1]
				if lastMsg.Role == "user" && len(lastMsg.Content) > 0 {
					if _, ok := lastMsg.Content[0].(anthropicToolResultBlock); ok {
						if i == len(opts.Messages)-1 {
							toolResult.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
						}
						lastMsg.Content = append(lastMsg.Content, toolResult)
						continue
					}
				}
			}

			// Cache tool results when they're the last message
			if i == len(opts.Messages)-1 {
				toolResult.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
			}
			antMsg.Content = []interface{}{toolResult}
		} else if msg.Role == "assistant" {
			// Assistant message - may have text and/or tool calls
			// Cache the last assistant message to avoid re-processing large tool_use blocks
			isLastAssistant := i == lastAssistantIdx
			hasToolCalls := len(msg.ToolCalls) > 0

			if msg.Content != "" {
				textBlock := anthropicTextBlock{
					Type: "text",
					Text: msg.Content,
				}
				// If this is the last assistant message and has no tool calls, cache the text block
				if isLastAssistant && !hasToolCalls {
					textBlock.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
				}
				antMsg.Content = append(antMsg.Content, textBlock)
			}
			for j, tc := range msg.ToolCalls {
				var input any
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
					return nil, fmt.Errorf("error parsing tool call arguments for %q: %w", tc.Function.Name, err)
				}
				toolUseBlock := anthropicToolUseBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				}
				// Cache the last tool_use block of the last assistant message
				if isLastAssistant && j == len(msg.ToolCalls)-1 {
					toolUseBlock.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
				}
				antMsg.Content = append(antMsg.Content, toolUseBlock)
			}
		} else {
			// User message - only cache the very last one
			isLastMessage := i == len(opts.Messages)-1

			if len(msg.MultiContent) > 0 {
				// Use explicit content blocks for interleaved text+image
				for _, block := range msg.MultiContent {
					switch block.Type {
					case "text":
						tb := anthropicTextBlock{
							Type: "text",
							Text: block.Text,
						}
						if block.Cache {
							tb.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
						}
						antMsg.Content = append(antMsg.Content, tb)
					case "image":
						if block.ImageURL != "" {
							antMsg.Content = append(antMsg.Content, anthropicImageBlock{
								Type: "image",
								Source: anthropicImageSource{
									Type: "url",
									URL:  block.ImageURL,
								},
							})
						} else if block.ImageBase64 != "" {
							mediaType := block.ImageMediaType
							if mediaType == "" {
								mediaType = DetectImageMediaType(block.ImageBase64)
							}
							antMsg.Content = append(antMsg.Content, anthropicImageBlock{
								Type: "image",
								Source: anthropicImageSource{
									Type:      "base64",
									MediaType: mediaType,
									Data:      block.ImageBase64,
								},
							})
						}
					case "document":
						antMsg.Content = append(antMsg.Content, buildAnthropicDocumentBlock(Document{
							Base64:    block.DocumentBase64,
							URL:       block.DocumentURL,
							MediaType: block.DocumentMediaType,
							Title:     block.DocumentTitle,
						}))
					}
				}
				// Add cache control to last block if this is the last message (fallback)
				if isLastMessage && len(antMsg.Content) > 0 {
					hasExplicitCache := false
					for _, block := range msg.MultiContent {
						if block.Cache {
							hasExplicitCache = true
							break
						}
					}
					if !hasExplicitCache {
						if tb, ok := antMsg.Content[len(antMsg.Content)-1].(anthropicTextBlock); ok {
							tb.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
							antMsg.Content[len(antMsg.Content)-1] = tb
						}
					}
				}
			} else {
				// Handle images
				for _, img := range msg.Images {
					antMsg.Content = append(antMsg.Content, anthropicImageBlock{
						Type: "image",
						Source: anthropicImageSource{
							Type:      "base64",
							MediaType: DetectImageMediaType(img),
							Data:      img,
						},
					})
				}

				// Handle documents (e.g. PDFs)
				for _, doc := range msg.Documents {
					antMsg.Content = append(antMsg.Content, buildAnthropicDocumentBlock(doc))
				}

				// Add text content
				textBlock := anthropicTextBlock{
					Type: "text",
					Text: msg.Content,
				}
				if isLastMessage {
					textBlock.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
				}
				antMsg.Content = append(antMsg.Content, textBlock)
			}
		}

		req.Messages = append(req.Messages, antMsg)
	}

	return req, nil
}

// parseAnthropicResponse converts an Anthropic API response into the standard ResponseMessageGenerate format.
// This is shared by both the direct Anthropic API and the Bedrock API paths.
func parseAnthropicResponse(resp *http.Response) (*ResponseMessageGenerate, error) {
	var antResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&antResp); err != nil {
		return nil, fmt.Errorf("error decoding Anthropic response: %w", err)
	}

	result := &ResponseMessageGenerate{
		Model: antResp.Model,
		Usage: Usage{
			PromptTokens:             antResp.Usage.InputTokens,
			CompletionTokens:         antResp.Usage.OutputTokens,
			TotalTokens:              antResp.Usage.InputTokens + antResp.Usage.OutputTokens,
			CacheCreationInputTokens: antResp.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     antResp.Usage.CacheReadInputTokens,
		},
		Choices: []GenChoice{
			{
				Index: 0,
				Message: Message{
					Role: antResp.Role,
				},
			},
		},
	}

	var toolCalls []ToolCall
	var textContent strings.Builder
	for _, block := range antResp.Content {
		switch block.Type {
		case "text":
			textContent.WriteString(block.Text)
		case "tool_use":
			inputJSON, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      block.Name,
					Arguments: string(inputJSON),
				},
			})
		}
	}

	result.Choices[0].Message.Content = textContent.String()
	result.Choices[0].Message.ToolCalls = toolCalls

	return result, nil
}

// ChatCompletionAnthropic sends a request using Anthropic's native API format with caching support.
// The system prompt and the last user message before each assistant turn are marked for caching.
func (c *Client) ChatCompletionAnthropic(opts RequestOptions) (*ResponseMessageGenerate, error) {
	req, err := buildAnthropicRequest(opts)
	if err != nil {
		return nil, err
	}

	// Send request to Anthropic's native endpoint
	resp, err := c.prepareRequest(req, c.anthropicEndpoint("/messages"))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return parseAnthropicResponse(resp)
}

// IsAnthropicAPI checks if the client is configured to use Anthropic's API.
// Returns true if SetAnthropicMode(true) was called, or if the base URL
// contains "anthropic.com" (auto-detection fallback).
func (c *Client) IsAnthropicAPI() bool {
	if c.anthropicMode != nil {
		return *c.anthropicMode
	}
	return strings.Contains(c.baseURL, "anthropic.com")
}
