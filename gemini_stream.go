package gollama

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// chatCompletionGeminiStream runs a Gemini streamGenerateContent turn over
// SSE. It fires onDelta with the full accumulated visible (non-thought) text
// — a snapshot, not an increment — whenever it grows, then converts the
// assembled response with convertGeminiResponse, the same converter the
// non-streaming path uses.
func (c *Client) chatCompletionGeminiStream(ctx context.Context, opts RequestOptions, onDelta func(text string)) (*ResponseMessageGenerate, error) {
	req := buildGeminiRequest(opts)
	// prepareRequestCtx handles pre-stream retryable statuses (429/503/529)
	// exactly like the non-streaming path; a 200 returns the open SSE body.
	resp, err := c.prepareRequestCtx(ctx, req, geminiModelPath(opts.Model)+":streamGenerateContent?alt=sse")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	assembled, err := assembleGeminiStream(resp.Body, onDelta, opts.OnGenerated)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return nil, fmt.Errorf("gemini stream: %w", cerr)
		}
		if ae, ok := AsAPIError(err); ok && len(ae.Header) == 0 {
			ae.Header = resp.Header.Clone() // embedded error: keep transport metadata
		}
		return nil, err
	}
	return convertGeminiResponse(assembled, opts.Model)
}

// assembleGeminiStream reads a streamGenerateContent SSE body, where every
// data line is a complete GenerateContentResponse carrying the next parts of
// candidate 0. Parts are concatenated in arrival order (Gemini sends each
// functionCall whole; a signature may arrive on a later signature-only part,
// which convertGeminiResponse attaches to the preceding call). Usage and
// modelVersion take the last value seen. The stream counts as complete only
// once a substantive candidate finishReason (not FINISH_REASON_UNSPECIFIED)
// or a real prompt blockReason arrives; EOF before that — including a
// connection cut midway through a JSON frame — is reported as an error
// wrapping io.ErrUnexpectedEOF so callers can retry. An error object embedded
// in a chunk is returned as an *APIError (see geminiEmbeddedAPIError).
//
// onGenerated (may be nil) fires once per chunk that carries generated output:
// a part with non-empty text (thought or visible) or a functionCall.
// Metadata-only chunks (usage, signature-only or empty parts) do not count.
func assembleGeminiStream(body io.Reader, onDelta func(text string), onGenerated func()) (*geminiResponse, error) {
	reader := bufio.NewReader(body)

	assembled := &geminiResponse{}
	var (
		content   geminiContent
		finish    string
		finishMsg string
		completed bool
		haveCand  bool
		visible   strings.Builder
	)

	for {
		line, rerr := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimRight(line, "\r\n")
			if data, ok := strings.CutPrefix(line, "data:"); ok {
				data = strings.TrimSpace(data)
				if data != "" && data != "[DONE]" {
					var chunk geminiResponse
					if err := json.Unmarshal([]byte(data), &chunk); err != nil {
						if rerr == io.EOF {
							// The body ended inside this frame: a truncated
							// stream, not a malformed one.
							return nil, fmt.Errorf("gemini stream cut mid-frame (%v): %w", err, io.ErrUnexpectedEOF)
						}
						if rerr != nil {
							// The read failed mid-frame (stall watchdog, ctx
							// cancellation, reset): preserve that cause.
							return nil, fmt.Errorf("error reading Gemini stream: %w", rerr)
						}
						return nil, fmt.Errorf("error decoding Gemini stream chunk: %w", err)
					}
					if chunk.Error != nil {
						return nil, geminiEmbeddedAPIError(chunk.Error, []byte(data), nil)
					}
					if chunk.ModelVersion != "" {
						assembled.ModelVersion = chunk.ModelVersion
					}
					if chunk.ResponseID != "" {
						assembled.ResponseID = chunk.ResponseID
					}
					if chunk.UsageMetadata != nil {
						assembled.UsageMetadata = chunk.UsageMetadata
					}
					if geminiBlocked(chunk.PromptFeedback) {
						assembled.PromptFeedback = chunk.PromptFeedback
						completed = true
					}
					if len(chunk.Candidates) > 0 {
						haveCand = true
						cand := chunk.Candidates[0]
						grew, generated := false, false
						if cand.Content != nil {
							if cand.Content.Role != "" {
								content.Role = cand.Content.Role
							}
							for _, p := range cand.Content.Parts {
								content.Parts = append(content.Parts, p)
								if p.Text != "" || p.FunctionCall != nil {
									generated = true
								}
								if !p.Thought && p.FunctionCall == nil && p.Text != "" {
									visible.WriteString(p.Text)
									grew = true
								}
							}
						}
						if generated && onGenerated != nil {
							onGenerated()
						}
						if grew && onDelta != nil {
							onDelta(visible.String())
						}
						if geminiTerminalFinish(cand.FinishReason) {
							finish = cand.FinishReason
							finishMsg = cand.FinishMessage
							completed = true
						}
					}
				}
			}
		}

		if rerr != nil {
			if rerr == io.EOF {
				break
			}
			return nil, fmt.Errorf("error reading Gemini stream: %w", rerr)
		}
	}

	if !completed {
		return nil, fmt.Errorf("gemini stream ended before finishReason: %w", io.ErrUnexpectedEOF)
	}
	if haveCand {
		assembled.Candidates = []geminiCandidate{{Content: &content, FinishReason: finish, FinishMessage: finishMsg}}
	}
	return assembled, nil
}
