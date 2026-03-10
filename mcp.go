package gollama

import (
	"context"
	"fmt"

	mcpgo "github.com/metoro-io/mcp-golang"
	http "github.com/metoro-io/mcp-golang/transport/http"
)

// ToolResult contains the result of a tool call, optionally with images
type ToolResult struct {
	Content string
	Images  []string // optional base64 encoded images
	IsError bool     // true if this result represents an error
}

type Tool struct {
	Name        string
	Description string
	Params      any

	Call func(context.Context, any) (*ToolResult, error)
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

	resp, err := client.Initialize(ctx)
	if err != nil {
		return nil, err
	}

	_ = resp

	tools, err := client.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}

	var out []*Tool
	for _, t := range tools.Tools {
		t := &Tool{
			Name:        t.Name,
			Description: *t.Description,
			Params:      t.InputSchema,
		}
		t.Call = mcpCall(t, client)
		out = append(out, t)

	}

	return out, nil
}
