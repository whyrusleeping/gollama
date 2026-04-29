package gollama

import (
	"context"
	"fmt"

	mcpgo "github.com/metoro-io/mcp-golang"
	http "github.com/metoro-io/mcp-golang/transport/http"
)

// ToolResult contains the result of a tool call.
// Optionally carries images and/or documents (e.g. PDFs) for backends that
// support them in tool result blocks (currently Anthropic native API only).
type ToolResult struct {
	Content    string
	Images     []string   // optional base64 encoded images
	Documents  []Document // optional documents (e.g. PDFs)
	IsError    bool       // true if this result represents an error
	Structured any        // optional structured form consumed by codemode/bridge paths; agent-loop path ignores it
}

type Tool struct {
	Name        string
	Description string
	Params      any
	OutputType  any // optional Go value whose reflected type describes the structured response shape (used by codemode typegen)

	Call func(context.Context, any) (*ToolResult, error)
}

// StringResultCall adapts a function returning (string, error) into the
// Tool.Call signature. The returned wrapper packs the string into
// ToolResult.Content. Useful when porting tools that don't need to attach
// images or documents.
func StringResultCall(fn func(context.Context, any) (string, error)) func(context.Context, any) (*ToolResult, error) {
	return func(ctx context.Context, params any) (*ToolResult, error) {
		s, err := fn(ctx, params)
		if err != nil {
			return nil, err
		}
		return &ToolResult{Content: s}, nil
	}
}

func (t *Tool) ApiDef() ToolParam {
	return ToolParam{
		Type: "function",
		Function: &ToolFunction{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Params,
		},
	}
}

func mcpCall(t *Tool, client *mcpgo.Client) func(ctx context.Context, param any) (*ToolResult, error) {
	return func(ctx context.Context, param any) (*ToolResult, error) {
		resp, err := client.CallTool(ctx, t.Name, param)
		if err != nil {
			return nil, err
		}

		var content string
		for _, c := range resp.Content {
			if c.TextContent != nil {
				content += fmt.Sprintf("%v", *c.TextContent)
			}
		}

		return &ToolResult{Content: content}, nil
	}
}

func ToolsFromMCP(ctx context.Context, host string) ([]*Tool, error) {
	tpt := http.NewHTTPClientTransport(host)
	client := mcpgo.NewClient(tpt)

	if _, err := client.Initialize(ctx); err != nil {
		return nil, err
	}

	tools, err := client.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}

	var out []*Tool
	for _, mt := range tools.Tools {
		desc := ""
		if mt.Description != nil {
			desc = *mt.Description
		}
		t := &Tool{
			Name:        mt.Name,
			Description: desc,
			Params:      mt.InputSchema,
		}
		t.Call = mcpCall(t, client)
		out = append(out, t)
	}

	return out, nil
}
