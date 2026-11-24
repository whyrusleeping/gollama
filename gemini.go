package gollama

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

// DefaultGeminiBaseURL is the Gemini API (AI Studio) REST base. Requests go to
// {base}/models/{model}:generateContent and :streamGenerateContent?alt=sse.
const DefaultGeminiBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// GeminiAPIKeyHeader carries a Gemini API key. Set it with
// SetHeader(GeminiAPIKeyHeader, key).
const GeminiAPIKeyHeader = "x-goog-api-key"

// geminiDummyThoughtSignature is Google's documented placeholder for a
// functionCall part that the API did not generate and so has no real thought
// signature (e.g. history produced by another model, or a synthetic call).
// Gemini 3+ validates that the first functionCall of every step in the
// current turn carries a signature and 400s otherwise; this value bypasses
// that check (at some cost to model quality, so real signatures always win).
// Source: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking/thought-signatures
// ("You can set thought_signature to skip_thought_signature_validator").
const geminiDummyThoughtSignature = "skip_thought_signature_validator"

// geminiSyntheticIDPrefix marks tool-call IDs gollama invents when Gemini
// omits functionCall.id. They are never sent back to Gemini (the API pairs
// calls and responses by name and order when ids are absent).
const geminiSyntheticIDPrefix = "gollama_call_"

// SetGeminiMode explicitly enables or disables the native Gemini API mode.
// By default, the client auto-detects Gemini endpoints from the base URL.
func (c *Client) SetGeminiMode(enabled bool) {
	c.geminiMode = &enabled
}

// IsGeminiAPI reports whether this client talks to the native Gemini API.
func (c *Client) IsGeminiAPI() bool {
	if c.geminiMode != nil {
		return *c.geminiMode
	}
	return strings.Contains(c.baseURL, "generativelanguage.googleapis.com") && !strings.Contains(c.baseURL, "/openai")
}

// Gemini generateContent wire types. Only the fields gollama reads or writes
// are modeled; unknown response fields are ignored for forward compatibility.

type geminiRequest struct {
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	Contents          []geminiContent         `json:"contents"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	ToolConfig        *geminiToolConfig       `json:"toolConfig,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"`
	InlineData       *geminiBlob             `json:"inlineData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiBlob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiFunctionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type geminiFunctionResponse struct {
	ID       string `json:"id,omitempty"`
	Name     string `json:"name"`
	Response any    `json:"response"`
	// Parts carries multimodal tool output (Gemini 3+): images/documents
	// returned by the tool travel inside the functionResponse rather than as
	// sibling parts, which keeps "all calls, then all responses" ordering.
	Parts []geminiFunctionResponsePart `json:"parts,omitempty"`
}

type geminiFunctionResponsePart struct {
	InlineData *geminiBlob `json:"inlineData,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

// geminiFunctionDeclaration uses parametersJsonSchema (full JSON Schema) rather
// than the legacy parameters field, which accepts only an OpenAPI 3.0 Schema
// subset and 400s on common keys such as additionalProperties or $schema. The
// two fields are mutually exclusive.
type geminiFunctionDeclaration struct {
	Name                 string `json:"name"`
	Description          string `json:"description,omitempty"`
	ParametersJSONSchema any    `json:"parametersJsonSchema,omitempty"`
}

type geminiToolConfig struct {
	FunctionCallingConfig geminiFunctionCallingConfig `json:"functionCallingConfig"`
}

type geminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type geminiGenerationConfig struct {
	MaxOutputTokens int                   `json:"maxOutputTokens,omitempty"`
	Temperature     *float64              `json:"temperature,omitempty"`
	TopP            *float64              `json:"topP,omitempty"`
	TopK            int                   `json:"topK,omitempty"`
	ThinkingConfig  *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

// geminiThinkingConfig deliberately has no thinkingBudget field: Gemini 3+
// models use thinkingLevel, and some (gemini-4-neon) reject a budget.
type geminiThinkingConfig struct {
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
	IncludeThoughts bool   `json:"includeThoughts,omitempty"`
}

type geminiResponse struct {
	Candidates     []geminiCandidate     `json:"candidates"`
	PromptFeedback *geminiPromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *geminiUsage          `json:"usageMetadata,omitempty"`
	ModelVersion   string                `json:"modelVersion,omitempty"`
	ResponseID     string                `json:"responseId,omitempty"`
	// Error appears on mid-stream failures of streamGenerateContent.
	Error *geminiError `json:"error,omitempty"`
}

type geminiCandidate struct {
	Content       *geminiContent `json:"content,omitempty"`
	FinishReason  string         `json:"finishReason,omitempty"`
	FinishMessage string         `json:"finishMessage,omitempty"`
	Index         int            `json:"index"`
}

type geminiPromptFeedback struct {
	BlockReason string `json:"blockReason,omitempty"`
}

type geminiUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
}

type geminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// geminiEmbeddedAPIError converts an error object Gemini delivered inside an
// HTTP 200 body (a streamed chunk, or a 200 generateContent body) into the
// same *APIError a non-200 response produces: StatusCode is the embedded
// error.code (500 when absent or out of range) and Body the raw JSON holding
// the error, so callers classify it, and parse RetryInfo, exactly as they
// would for a real HTTP error.
func geminiEmbeddedAPIError(e *geminiError, raw []byte, header http.Header) *APIError {
	code := e.Code
	if code < 400 || code > 599 {
		code = http.StatusInternalServerError
	}
	if header == nil {
		header = http.Header{}
	}
	return &APIError{StatusCode: code, Body: string(raw), Header: header.Clone()}
}

// geminiFinishAPIError reports a finishReason that means the turn failed
// rather than completed, as an *APIError with a Google-style error body.
func geminiFinishAPIError(code int, status, reason, message string) *APIError {
	msg := "gemini finishReason " + reason
	if message != "" {
		msg += ": " + message
	}
	body, _ := json.Marshal(map[string]any{"error": map[string]any{"code": code, "message": msg, "status": status}})
	return &APIError{StatusCode: code, Body: string(body), Header: http.Header{}}
}

// geminiBlocked reports whether promptFeedback carries a real block reason.
func geminiBlocked(pf *geminiPromptFeedback) bool {
	return pf != nil && pf.BlockReason != "" && pf.BlockReason != "BLOCK_REASON_UNSPECIFIED"
}

// geminiTerminalFinish reports whether a finishReason ends the candidate
// (the docs: "If empty, the model has not stopped generating tokens").
func geminiTerminalFinish(reason string) bool {
	return reason != "" && reason != "FINISH_REASON_UNSPECIFIED"
}

// geminiModelPath returns the "/models/{model}" path for a model name,
// accepting both "gemini-x" and the API's "models/gemini-x" spelling.
func geminiModelPath(model string) string {
	return "/models/" + strings.TrimPrefix(model, "models/")
}

// mapGeminiThinkingLevel translates a gollama Effort level to a Gemini
// thinkingLevel. "max" clamps to XHIGH. MINIMAL is never produced (some
// Gemini 4 models reject it); unknown or empty values return "" so the
// model's default applies.
func mapGeminiThinkingLevel(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "low":
		return "LOW"
	case "medium":
		return "MEDIUM"
	case "high":
		return "HIGH"
	case "xhigh", "max":
		return "XHIGH"
	}
	return ""
}

// geminiArgs converts a ToolCall's JSON-string Arguments into the object
// Gemini requires for functionCall.args. Replay must never fail on a
// malformed historical call, so invalid or non-object JSON is wrapped as
// {"_raw": "<original>"}.
func geminiArgs(arguments string) json.RawMessage {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return json.RawMessage(`{}`)
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(trimmed), &obj); err == nil && obj != nil {
		return json.RawMessage(trimmed)
	}
	raw, _ := json.Marshal(map[string]string{"_raw": arguments})
	return raw
}

// geminiWireID returns the id to send for a call/response pair, dropping ids
// gollama synthesized itself (Gemini never saw them).
func geminiWireID(id string) string {
	if strings.HasPrefix(id, geminiSyntheticIDPrefix) {
		return ""
	}
	return id
}

func newGeminiCallID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("gollama: crypto/rand failed: %v", err))
	}
	return geminiSyntheticIDPrefix + hex.EncodeToString(b[:])
}

// geminiImagePart converts an image into an inline part. URL-only media is
// not fetched (Gemini's fileData takes Files API / Cloud Storage URIs, not
// arbitrary web URLs), so it is referenced as text instead.
func geminiImagePart(b64, mediaType, imageURL string) geminiPart {
	if b64 == "" {
		return geminiPart{Text: "[image: " + imageURL + "]"}
	}
	if mediaType == "" {
		mediaType = DetectImageMediaType(b64)
	}
	return geminiPart{InlineData: &geminiBlob{MimeType: mediaType, Data: b64}}
}

// geminiDocumentPart converts a document into an inline part (default
// application/pdf); URL-only documents are referenced as text.
func geminiDocumentPart(doc Document) geminiPart {
	if doc.Base64 == "" {
		label := doc.URL
		if doc.Title != "" {
			label = doc.Title + " (" + doc.URL + ")"
		}
		return geminiPart{Text: "[document: " + label + "]"}
	}
	mediaType := doc.MediaType
	if mediaType == "" {
		mediaType = "application/pdf"
	}
	return geminiPart{InlineData: &geminiBlob{MimeType: mediaType, Data: doc.Base64}}
}

// buildGeminiRequest converts generic RequestOptions into a generateContent
// request body. It is shared by the streaming and non-streaming paths.
func buildGeminiRequest(opts RequestOptions) *geminiRequest {
	req := &geminiRequest{Contents: []geminiContent{}}

	// System instruction. Priority mirrors the Anthropic path: SystemBlocks >
	// System > role="system" messages. System messages are never sent as
	// contents (Gemini has only user/model roles).
	var sysParts []geminiPart
	switch {
	case len(opts.SystemBlocks) > 0:
		for _, b := range opts.SystemBlocks {
			if b.Text != "" {
				sysParts = append(sysParts, geminiPart{Text: b.Text})
			}
		}
	case opts.System != "":
		sysParts = []geminiPart{{Text: opts.System}}
	default:
		var texts []string
		for _, m := range opts.Messages {
			if m.Role == "system" && m.Content != "" {
				texts = append(texts, m.Content)
			}
		}
		if len(texts) > 0 {
			sysParts = []geminiPart{{Text: strings.Join(texts, "\n\n")}}
		}
	}
	if len(sysParts) > 0 {
		req.SystemInstruction = &geminiContent{Parts: sysParts}
	}

	// Tools.
	var decls []geminiFunctionDeclaration
	declared := map[string]bool{}
	for _, t := range opts.Tools {
		if t.Function == nil {
			continue
		}
		decls = append(decls, geminiFunctionDeclaration{
			Name:                 t.Function.Name,
			Description:          t.Function.Description,
			ParametersJSONSchema: prepareGeminiSchema(t.Function.Parameters),
		})
		declared[t.Function.Name] = true
	}
	if len(decls) > 0 {
		req.Tools = []geminiTool{{FunctionDeclarations: decls}}
	}
	mode := func(m string, allowed ...string) *geminiToolConfig {
		return &geminiToolConfig{FunctionCallingConfig: geminiFunctionCallingConfig{Mode: m, AllowedFunctionNames: allowed}}
	}
	switch tc := strings.ToLower(strings.TrimSpace(opts.ToolChoice)); {
	case tc == "":
	case tc == "auto":
		req.ToolConfig = mode("AUTO")
	case tc == "none":
		req.ToolConfig = mode("NONE")
	case tc == "required" || tc == "any":
		req.ToolConfig = mode("ANY")
	case declared[opts.ToolChoice]:
		req.ToolConfig = mode("ANY", opts.ToolChoice)
	}

	// Generation config.
	gc := &geminiGenerationConfig{}
	if o := opts.Options; o != nil {
		if o.MaxTokens > 0 {
			gc.MaxOutputTokens = o.MaxTokens
		}
		if o.Temperature != 0 {
			t := o.Temperature
			gc.Temperature = &t
		}
		if o.TopP != 0 {
			p := o.TopP
			gc.TopP = &p
		}
		if o.TopK > 0 {
			gc.TopK = o.TopK
		}
	}
	th := &geminiThinkingConfig{
		ThinkingLevel:   mapGeminiThinkingLevel(opts.Effort),
		IncludeThoughts: opts.ThinkingDisplay != "omitted",
	}
	if th.ThinkingLevel != "" || th.IncludeThoughts {
		gc.ThinkingConfig = th
	}
	if *gc != (geminiGenerationConfig{}) {
		req.GenerationConfig = gc
	}

	// Contents. Gemini requires user/model alternation, so consecutive
	// same-role contents (e.g. parallel tool results, or a tool result followed
	// by a user interjection) are merged into one content.
	appendContent := func(role string, parts []geminiPart) {
		if len(parts) == 0 {
			return
		}
		if n := len(req.Contents); n > 0 && req.Contents[n-1].Role == role {
			req.Contents[n-1].Parts = append(req.Contents[n-1].Parts, parts...)
			return
		}
		req.Contents = append(req.Contents, geminiContent{Role: role, Parts: parts})
	}

	// Tool results are paired with the calls of the latest assistant message.
	// A run of consecutive results is resolved as a batch: explicit ids are
	// matched first across the whole batch, then id-less, unknown-id or
	// duplicate-id results fill the remaining unanswered calls in call order.
	// A paired result takes BOTH its wire id and name from its call, and the
	// batch is emitted in call order, so out-of-order results for id-less
	// parallel calls to the same function cannot swap and no call is answered
	// twice. Results left without a call (e.g. answering an earlier turn's
	// call) go last in arrival order.
	callNames := map[string]string{} // tool call id -> function name (all turns)
	var pending []ToolCall           // calls of the latest assistant message
	var answered []bool              // answered[i] reports whether pending[i] has a result
	var batch []Message              // consecutive tool results awaiting resolution
	inPending := func(id string) bool {
		for _, p := range pending {
			if p.ID == id {
				return true
			}
		}
		return false
	}
	flushResults := func() {
		if len(batch) == 0 {
			return
		}
		call := make([]int, len(batch)) // index into pending, or -1
		for i := range call {
			call[i] = -1
		}
		// Pass 1: explicit ids of unanswered calls.
		for i, r := range batch {
			if r.ToolCallID == "" {
				continue
			}
			for j, p := range pending {
				if !answered[j] && p.ID == r.ToolCallID {
					call[i], answered[j] = j, true
					break
				}
			}
		}
		// Pass 2: the rest fill the remaining calls in order, except results
		// whose id names an earlier turn's call.
		next := 0
		for i, r := range batch {
			if call[i] >= 0 {
				continue
			}
			if _, older := callNames[r.ToolCallID]; older && !inPending(r.ToolCallID) {
				continue
			}
			for next < len(pending) && answered[next] {
				next++
			}
			if next < len(pending) {
				call[i], answered[next] = next, true
			}
		}

		type entry struct {
			order int
			fr    *geminiFunctionResponse
		}
		entries := make([]entry, len(batch))
		used := map[string]bool{}
		for i, r := range batch {
			fr := &geminiFunctionResponse{Response: map[string]any{"result": r.Content}}
			if j := call[i]; j >= 0 {
				fr.ID, fr.Name = geminiWireID(pending[j].ID), pending[j].Function.Name
				entries[i].order = j
				used[fr.ID] = true
			} else {
				fr.ID, fr.Name = geminiWireID(r.ToolCallID), callNames[r.ToolCallID]
				entries[i].order = len(pending) + i
			}
			if fr.Name == "" {
				fr.Name = "unknown_tool"
			}
			for _, img := range r.Images {
				fr.Parts = append(fr.Parts, geminiFunctionResponsePart{InlineData: &geminiBlob{MimeType: DetectImageMediaType(img), Data: img}})
			}
			for _, doc := range r.Documents {
				if p := geminiDocumentPart(doc); p.InlineData != nil {
					fr.Parts = append(fr.Parts, geminiFunctionResponsePart{InlineData: p.InlineData})
				}
			}
			entries[i].fr = fr
		}
		// Leftovers never repeat an id already answered in this batch.
		for i := range entries {
			if call[i] < 0 && entries[i].fr.ID != "" {
				if used[entries[i].fr.ID] {
					entries[i].fr.ID = ""
				}
				used[entries[i].fr.ID] = true
			}
		}
		sort.SliceStable(entries, func(i, j int) bool { return entries[i].order < entries[j].order })
		parts := make([]geminiPart, len(entries))
		for i, e := range entries {
			parts[i] = geminiPart{FunctionResponse: e.fr}
		}
		batch = nil
		appendContent("user", parts)
	}

	for _, msg := range opts.Messages {
		if msg.Role != "tool" && msg.Role != "system" {
			flushResults()
		}
		switch msg.Role {
		case "system":
			continue
		case "assistant":
			var parts []geminiPart
			// Thought-summary text is not replayed: Gemini restores reasoning
			// from signatures, and summaries are output-only. Signatures that
			// Gemini attaches to plain text parts are also dropped (the docs
			// call returning them recommended but optional); only functionCall
			// signatures are required and they round-trip via
			// ToolCall.ThoughtSignature.
			if msg.Content != "" {
				parts = append(parts, geminiPart{Text: msg.Content})
			}
			pending, answered = nil, nil
			for _, tc := range msg.ToolCalls {
				parts = append(parts, geminiPart{
					FunctionCall: &geminiFunctionCall{
						ID:   geminiWireID(tc.ID),
						Name: tc.Function.Name,
						Args: geminiArgs(tc.Function.Arguments),
					},
					ThoughtSignature: tc.ThoughtSignature,
				})
				if tc.ID != "" {
					callNames[tc.ID] = tc.Function.Name
				}
				pending = append(pending, tc)
				answered = append(answered, false)
			}
			appendContent("model", parts)
		case "tool":
			batch = append(batch, msg)
		default: // "user"
			var parts []geminiPart
			if len(msg.MultiContent) > 0 {
				for _, b := range msg.MultiContent {
					switch b.Type {
					case "text":
						if b.Text != "" {
							parts = append(parts, geminiPart{Text: b.Text})
						}
					case "image":
						if b.ImageBase64 != "" || b.ImageURL != "" {
							parts = append(parts, geminiImagePart(b.ImageBase64, b.ImageMediaType, b.ImageURL))
						}
					case "document":
						if b.DocumentBase64 != "" || b.DocumentURL != "" {
							parts = append(parts, geminiDocumentPart(Document{
								Base64:    b.DocumentBase64,
								URL:       b.DocumentURL,
								MediaType: b.DocumentMediaType,
								Title:     b.DocumentTitle,
							}))
						}
					}
				}
			} else {
				for _, img := range msg.Images {
					parts = append(parts, geminiImagePart(img, "", ""))
				}
				for _, doc := range msg.Documents {
					parts = append(parts, geminiDocumentPart(doc))
				}
				if msg.Content != "" {
					parts = append(parts, geminiPart{Text: msg.Content})
				}
			}
			appendContent("user", parts)
		}
	}

	flushResults()
	applyGeminiDummySignatures(req.Contents)
	return req
}

// applyGeminiDummySignatures enforces Gemini's thought-signature rule for the
// current turn. Per Google's docs, "a turn begins with the most recent user
// message that is not a functionResponse", and within it "omitting a required
// thought_signature for the first functionCall part in any step of the current
// turn results in a 400 error" (parallel calls carry a signature only on the
// first call). Steps whose first call has none — another model's calls after
// a model switch, synthetic calls — get Google's documented placeholder.
// Earlier turns are not validated and are left untouched.
func applyGeminiDummySignatures(contents []geminiContent) {
	start := 0
	for i := len(contents) - 1; i >= 0; i-- {
		if contents[i].Role != "user" {
			continue
		}
		hasNonResponse := false
		for _, p := range contents[i].Parts {
			if p.FunctionResponse == nil {
				hasNonResponse = true
				break
			}
		}
		if hasNonResponse {
			start = i + 1
			break
		}
	}
	for i := start; i < len(contents); i++ {
		if contents[i].Role != "model" {
			continue
		}
		for j := range contents[i].Parts {
			p := &contents[i].Parts[j]
			if p.FunctionCall == nil {
				continue
			}
			if p.ThoughtSignature == "" {
				p.ThoughtSignature = geminiDummyThoughtSignature
			}
			break // only the first functionCall of a step is validated
		}
	}
}

// prepareGeminiSchema converts a tool's parameters into a JSON Schema object
// for parametersJsonSchema. The schema is normalized via a JSON round trip,
// then: the root "$schema" key is dropped (Gemini rejects it), null-valued
// keywords such as a nil "required" are removed, same-document "$ref"s are
// inlined (Gemini does not reliably resolve reference indirection; on a
// circular or unresolvable ref the schema is sent as-is), and an object root
// is guaranteed to have "properties".
func prepareGeminiSchema(params any) map[string]any {
	empty := func() map[string]any {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	if params == nil {
		return empty()
	}
	data, err := json.Marshal(params)
	if err != nil {
		return empty()
	}
	var schema map[string]any
	if err := json.Unmarshal(data, &schema); err != nil || len(schema) == 0 {
		return empty()
	}
	delete(schema, "$schema")
	dropNullSchemaKeys(schema)

	if hasSchemaRef(schema) {
		budget := 256
		if inlined, err := inlineSchemaRefs(schema, schema, &budget, nil); err == nil {
			if m, ok := inlined.(map[string]any); ok {
				schema = m
				delete(schema, "$defs")
				delete(schema, "definitions")
			}
		}
	}
	if len(schema) == 0 {
		return empty()
	}
	if schema["type"] == "object" {
		if _, ok := schema["properties"]; !ok {
			schema["properties"] = map[string]any{}
		}
	}
	return schema
}

// dropNullSchemaKeys removes null-valued keywords in place (Go nil slices and
// maps marshal as null, e.g. ToolFunctionParams.Required), keeping keywords
// where null is a meaningful value.
func dropNullSchemaKeys(node any) {
	switch v := node.(type) {
	case map[string]any:
		for k, val := range v {
			if val == nil {
				switch k {
				case "const", "default", "example", "examples":
				default:
					delete(v, k)
				}
				continue
			}
			dropNullSchemaKeys(val)
		}
	case []any:
		for _, item := range v {
			dropNullSchemaKeys(item)
		}
	}
}

func hasSchemaRef(node any) bool {
	switch v := node.(type) {
	case map[string]any:
		if _, ok := v["$ref"].(string); ok {
			return true
		}
		for _, val := range v {
			if hasSchemaRef(val) {
				return true
			}
		}
	case []any:
		for _, val := range v {
			if hasSchemaRef(val) {
				return true
			}
		}
	}
	return false
}

// inlineSchemaRefs returns a copy of node with same-document $refs replaced by
// their targets. It fails on circular or unresolvable refs and when the
// expansion budget runs out, so the caller can fall back to the original.
func inlineSchemaRefs(node any, root map[string]any, budget *int, stack []string) (any, error) {
	switch v := node.(type) {
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			r, err := inlineSchemaRefs(item, root, budget, stack)
			if err != nil {
				return nil, err
			}
			out[i] = r
		}
		return out, nil
	case map[string]any:
		ref, isRef := v["$ref"].(string)
		if !isRef {
			out := make(map[string]any, len(v))
			for k, item := range v {
				r, err := inlineSchemaRefs(item, root, budget, stack)
				if err != nil {
					return nil, err
				}
				out[k] = r
			}
			return out, nil
		}
		for _, s := range stack {
			if s == ref {
				return nil, fmt.Errorf("circular $ref %q", ref)
			}
		}
		*budget--
		if *budget < 0 {
			return nil, fmt.Errorf("$ref expansion budget exhausted")
		}
		target := resolveLocalSchemaRef(root, ref)
		if target == nil {
			return nil, fmt.Errorf("unresolvable $ref %q", ref)
		}
		inlined, err := inlineSchemaRefs(target, root, budget, append(stack[:len(stack):len(stack)], ref))
		if err != nil {
			return nil, err
		}
		out := inlined.(map[string]any)
		// Sibling keywords of $ref (description, default, ...) win.
		for k, item := range v {
			if k == "$ref" {
				continue
			}
			r, err := inlineSchemaRefs(item, root, budget, stack)
			if err != nil {
				return nil, err
			}
			out[k] = r
		}
		return out, nil
	}
	return node, nil
}

func resolveLocalSchemaRef(root map[string]any, ref string) map[string]any {
	if !strings.HasPrefix(ref, "#/") {
		return nil
	}
	var node any = root
	for _, raw := range strings.Split(ref[2:], "/") {
		part := strings.ReplaceAll(strings.ReplaceAll(raw, "~1", "/"), "~0", "~")
		m, ok := node.(map[string]any)
		if !ok {
			return nil
		}
		if node, ok = m[part]; !ok {
			return nil
		}
	}
	m, _ := node.(map[string]any)
	return m
}

// normalizeGeminiFinishReason maps a documented Gemini FinishReason
// (https://ai.google.dev/api/generate-content#FinishReason) to gollama's
// OpenAI-style vocabulary, or to an error for reasons that mean the turn
// failed:
//
//   - STOP -> "stop"; MAX_TOKENS and CONTINUATION (per-request token limit,
//     generation incomplete) -> "length", so Truncated works.
//   - Content/policy filters (SAFETY, RECITATION, LANGUAGE, BLOCKLIST,
//     PROHIBITED_CONTENT, SPII, ESCALATION, IMAGE_SAFETY,
//     IMAGE_PROHIBITED_CONTENT, IMAGE_RECITATION) -> "refusal".
//   - Generation failures (MALFORMED_FUNCTION_CALL, MALFORMED_RESPONSE,
//     UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS) -> retryable 500 INTERNAL
//     *APIError: the model's output was unusable, and a retry usually works.
//   - MISSING_THOUGHT_SIGNATURE -> non-retryable 400 INVALID_ARGUMENT
//     *APIError (the replayed history is wrong; retrying cannot help).
//   - PUP_LIMITED_DISABLED -> 403 PERMISSION_DENIED *APIError (account
//     limited/disabled).
//   - Anything else (OTHER, IMAGE_OTHER, NO_IMAGE, future values) is
//     lowercased and passed through.
func normalizeGeminiFinishReason(reason, message string) (string, error) {
	switch reason {
	case "", "FINISH_REASON_UNSPECIFIED":
		return "", nil
	case "STOP":
		return "stop", nil
	case "MAX_TOKENS", "CONTINUATION":
		return "length", nil
	case "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "ESCALATION",
		"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return "refusal", nil
	case "MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS":
		return "", geminiFinishAPIError(http.StatusInternalServerError, "INTERNAL", reason, message)
	case "MISSING_THOUGHT_SIGNATURE":
		return "", geminiFinishAPIError(http.StatusBadRequest, "INVALID_ARGUMENT", reason, message)
	case "PUP_LIMITED_DISABLED":
		return "", geminiFinishAPIError(http.StatusForbidden, "PERMISSION_DENIED", reason, message)
	}
	return strings.ToLower(reason), nil
}

// convertGeminiResponse normalizes a generateContent response (or a streamed
// response assembled into the same shape) into ResponseMessageGenerate. It is
// shared by the streaming and non-streaming paths so both produce the same
// message for the same output (apart from synthesized tool-call IDs).
func convertGeminiResponse(gr *geminiResponse, model string) (*ResponseMessageGenerate, error) {
	if gr.ModelVersion != "" {
		model = gr.ModelVersion
	}
	result := &ResponseMessageGenerate{Model: model}
	if u := gr.UsageMetadata; u != nil {
		result.Usage = Usage{
			PromptTokens:        u.PromptTokenCount,
			CompletionTokens:    u.CandidatesTokenCount + u.ThoughtsTokenCount,
			TotalTokens:         u.TotalTokenCount,
			PromptTokensDetails: &PromptTokensDetails{CachedTokens: u.CachedContentTokenCount},
		}
	}

	msg := Message{Role: "assistant"}
	var finish string
	switch {
	case len(gr.Candidates) > 0:
		cand := gr.Candidates[0]
		var err error
		if finish, err = normalizeGeminiFinishReason(cand.FinishReason, cand.FinishMessage); err != nil {
			return nil, err
		}
		var text, thinking strings.Builder
		lastCall := -1 // index into msg.ToolCalls when the preceding part was a call
		if cand.Content != nil {
			for _, p := range cand.Content.Parts {
				switch {
				case p.FunctionCall != nil:
					args := "{}"
					if len(p.FunctionCall.Args) > 0 && string(p.FunctionCall.Args) != "null" {
						args = string(p.FunctionCall.Args)
					}
					id := p.FunctionCall.ID
					if id == "" {
						id = newGeminiCallID()
					}
					msg.ToolCalls = append(msg.ToolCalls, ToolCall{
						ID:               id,
						Type:             "function",
						Function:         ToolCallFunction{Name: p.FunctionCall.Name, Arguments: args},
						ThoughtSignature: p.ThoughtSignature,
					})
					lastCall = len(msg.ToolCalls) - 1
				case p.Thought:
					thinking.WriteString(p.Text)
					lastCall = -1
				case p.Text != "":
					text.WriteString(p.Text)
					lastCall = -1
				case p.ThoughtSignature != "":
					// A signature-only part (streaming may deliver it after
					// the part it belongs to). Attach it to an unsigned call
					// right before it; otherwise it is a text-part signature,
					// optional on replay, and dropped.
					if lastCall >= 0 && msg.ToolCalls[lastCall].ThoughtSignature == "" {
						msg.ToolCalls[lastCall].ThoughtSignature = p.ThoughtSignature
					}
				}
			}
		}
		msg.Content = text.String()
		msg.Thinking = thinking.String()
	case geminiBlocked(gr.PromptFeedback):
		// The prompt itself was blocked and no candidate was produced:
		// report a refusal stop with empty content, not an error.
		finish = "refusal"
	default:
		return nil, fmt.Errorf("gemini: response has no candidates")
	}

	// A refusal-class finish ends the turn: calls emitted before the filter
	// fired must not be dispatched (text and usage are kept).
	if finish == "refusal" {
		msg.ToolCalls = nil
	}
	// Tool calls mark a clean stop as "tool_calls", but never mask a
	// truncation or refusal: a MAX_TOKENS turn keeps "length" (and its calls)
	// so Truncated reports it.
	if len(msg.ToolCalls) > 0 && (finish == "stop" || finish == "") {
		finish = "tool_calls"
	}
	result.StopReason = finish
	result.Choices = []GenChoice{{Index: 0, Message: msg, FinishReason: finish}}
	return result, nil
}

func (c *Client) chatCompletionGemini(ctx context.Context, opts RequestOptions) (*ResponseMessageGenerate, error) {
	req := buildGeminiRequest(opts)
	resp, err := c.prepareRequestCtx(ctx, req, geminiModelPath(opts.Model)+":generateContent")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading Gemini response: %w", err)
	}
	var gr geminiResponse
	if err := json.Unmarshal(raw, &gr); err != nil {
		return nil, fmt.Errorf("error decoding Gemini response: %w", err)
	}
	if gr.Error != nil {
		return nil, geminiEmbeddedAPIError(gr.Error, raw, resp.Header)
	}
	return convertGeminiResponse(&gr, opts.Model)
}

type geminiModel struct {
	Name                       string   `json:"name"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
}

type geminiListModelsResponse struct {
	Models        []geminiModel `json:"models"`
	NextPageToken string        `json:"nextPageToken"`
}

// listGeminiModels pages through GET {base}/models and returns the models
// that support generateContent, with the "models/" prefix stripped from IDs.
func (c *Client) listGeminiModels(ctx context.Context) ([]ModelDesc, error) {
	var out []ModelDesc
	token := ""
	seen := map[string]bool{}
	for page := 0; page < 100; page++ {
		endpoint := "/models?pageSize=1000"
		if token != "" {
			endpoint += "&pageToken=" + url.QueryEscape(token)
		}
		resp, err := c.prepareGetCtx(ctx, endpoint)
		if err != nil {
			return nil, err
		}
		var lr geminiListModelsResponse
		err = json.NewDecoder(resp.Body).Decode(&lr)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("error decoding Gemini models: %w", err)
		}
		for _, m := range lr.Models {
			for _, method := range m.SupportedGenerationMethods {
				if method == "generateContent" {
					out = append(out, ModelDesc{ID: strings.TrimPrefix(m.Name, "models/"), Object: "model"})
					break
				}
			}
		}
		if lr.NextPageToken == "" || seen[lr.NextPageToken] {
			break
		}
		seen[lr.NextPageToken] = true
		token = lr.NextPageToken
	}
	return out, nil
}
