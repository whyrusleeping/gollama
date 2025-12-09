package gollama

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolParam represents a tool parameter in API requests.
// Used to define available tools/functions that the model can call.
type ToolParam struct {
	Type     string        `json:"type"`
	Function *ToolFunction `json:"function,omitempty"`
}

// ToolFunction defines a callable function/tool with its parameters.
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`

	// Parameters is usually satisfied by ToolFunctionParams but left as an
	// any for types from the MCP library to be used directly
	Parameters any `json:"parameters"`
}

// ToolFunctionParams defines the JSON schema for a tool's parameters.
type ToolFunctionParams struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required"`
}

// ToolCall represents a tool/function call made by the model.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
	Type     string           `json:"type"`
	ID       string           `json:"id"`
}

// ToolCallFunction contains the name and arguments of a called tool.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// HandleToolCall finds and executes a tool by name from the given tool list.
// Returns the tool's response string or an error.
func HandleToolCall(ctx context.Context, tools []*Tool, call ToolCall) (string, error) {
	var tool *Tool
	for _, t := range tools {
		if t.Name == call.Function.Name {
			tool = t
			break
		}
	}
	if tool == nil {
		return "", fmt.Errorf("no such tool %q", call.Function.Name)
	}

	var params map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
		return "", fmt.Errorf("invalid tool arguments: %w", err)
	}

	return tool.Call(ctx, params)
}
