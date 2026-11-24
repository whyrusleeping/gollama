package gollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func geminiTestClient(url string) *Client {
	c := NewClient(url)
	c.SetGeminiMode(true)
	c.SetHeader(GeminiAPIKeyHeader, "test-key")
	c.SetMaxRetries(0)
	return c
}

// geminiRequestJSON builds the request for opts and decodes it generically so
// tests assert on the wire shape rather than on Go structs.
func geminiRequestJSON(t *testing.T, opts RequestOptions) map[string]any {
	t.Helper()
	data, err := json.Marshal(buildGeminiRequest(opts))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

func jsonEqual(t *testing.T, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("bad want JSON: %v", err)
	}
	gb, _ := json.Marshal(got)
	var g any
	json.Unmarshal(gb, &g)
	if !reflect.DeepEqual(g, w) {
		t.Fatalf("JSON mismatch\n got: %s\nwant: %s", gb, want)
	}
}

func TestGeminiBackendDetection(t *testing.T) {
	if b := NewClient(DefaultGeminiBaseURL).Backend(); b != BackendGemini {
		t.Fatalf("Backend() = %v, want gemini", b)
	}
	if NewClient("https://generativelanguage.googleapis.com/v1beta/openai").IsGeminiAPI() {
		t.Fatal("OpenAI-compatible Gemini endpoint must not use the native backend")
	}
}

func TestGeminiRequestConfigAndTools(t *testing.T) {
	req := geminiRequestJSON(t, RequestOptions{
		Model:        "gemini-4-neon",
		SystemBlocks: []SystemBlock{{Text: "static", Cache: true}, {Text: "dynamic"}},
		Messages: []Message{
			{Role: "system", Content: "ignored: SystemBlocks win"},
			{Role: "user", Content: "hi"},
		},
		Tools: []ToolParam{
			{Type: "function", Function: &ToolFunction{
				Name: "read", Description: "Read a file",
				Parameters: ToolFunctionParams{Type: "object", Properties: map[string]any{"path": map[string]any{"type": "string"}}},
			}},
			{Type: "function", Function: &ToolFunction{
				Name: "noargs",
			}},
			{Type: "function", Function: &ToolFunction{
				Name: "refs",
				Parameters: map[string]any{
					"$schema":              "http://json-schema.org/draft-07/schema#",
					"type":                 "object",
					"additionalProperties": false,
					"properties":           map[string]any{"item": map[string]any{"$ref": "#/$defs/Item", "description": "the item"}},
					"$defs":                map[string]any{"Item": map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}}},
				},
			}},
		},
		ToolChoice:      "required",
		Options:         &Options{MaxTokens: 1234, Temperature: 0.5},
		Effort:          "max",
		ThinkingDisplay: "summarized",
	})

	jsonEqual(t, req["systemInstruction"], `{"parts":[{"text":"static"},{"text":"dynamic"}]}`)
	jsonEqual(t, req["contents"], `[{"role":"user","parts":[{"text":"hi"}]}]`)
	jsonEqual(t, req["tools"], `[{"functionDeclarations":[
		{"name":"read","description":"Read a file","parametersJsonSchema":{"type":"object","properties":{"path":{"type":"string"}}}},
		{"name":"noargs","parametersJsonSchema":{"type":"object","properties":{}}},
		{"name":"refs","parametersJsonSchema":{"type":"object","additionalProperties":false,
			"properties":{"item":{"type":"object","description":"the item","properties":{"name":{"type":"string"}}}}}}
	]}]`)
	jsonEqual(t, req["toolConfig"], `{"functionCallingConfig":{"mode":"ANY"}}`)
	jsonEqual(t, req["generationConfig"], `{"maxOutputTokens":1234,"temperature":0.5,"thinkingConfig":{"thinkingLevel":"XHIGH","includeThoughts":true}}`)
}

func TestGeminiThinkingLevelMapping(t *testing.T) {
	cases := []struct {
		effort, display string
		want            string // generationConfig JSON, "" = absent
	}{
		{"", "", `{"thinkingConfig":{"includeThoughts":true}}`},
		{"low", "", `{"thinkingConfig":{"thinkingLevel":"LOW","includeThoughts":true}}`},
		{"medium", "", `{"thinkingConfig":{"thinkingLevel":"MEDIUM","includeThoughts":true}}`},
		{"high", "", `{"thinkingConfig":{"thinkingLevel":"HIGH","includeThoughts":true}}`},
		{"xhigh", "", `{"thinkingConfig":{"thinkingLevel":"XHIGH","includeThoughts":true}}`},
		{"max", "", `{"thinkingConfig":{"thinkingLevel":"XHIGH","includeThoughts":true}}`},
		{"minimal", "", `{"thinkingConfig":{"includeThoughts":true}}`}, // never MINIMAL
		{"turbo", "", `{"thinkingConfig":{"includeThoughts":true}}`},   // unknown: omitted
		{"MAX", "", `{"thinkingConfig":{"thinkingLevel":"XHIGH","includeThoughts":true}}`},
		{"high", "omitted", `{"thinkingConfig":{"thinkingLevel":"HIGH"}}`},
		{"", "omitted", ""},
	}
	for _, tc := range cases {
		req := geminiRequestJSON(t, RequestOptions{Model: "m", Effort: tc.effort, ThinkingDisplay: tc.display,
			Messages: []Message{{Role: "user", Content: "x"}}})
		gc, ok := req["generationConfig"]
		if tc.want == "" {
			if ok {
				t.Errorf("effort=%q display=%q: unexpected generationConfig %v", tc.effort, tc.display, gc)
			}
			continue
		}
		jsonEqual(t, gc, tc.want)
		if strings.Contains(mustJSON(gc), "thinkingBudget") {
			t.Errorf("thinkingBudget must never be sent")
		}
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestGeminiRequestHistory(t *testing.T) {
	call := func(id, name, args, sig string) ToolCall {
		return ToolCall{ID: id, Type: "function", Function: ToolCallFunction{Name: name, Arguments: args}, ThoughtSignature: sig}
	}
	req := geminiRequestJSON(t, RequestOptions{
		Model:  "gemini-4-neon",
		System: "be brief",
		Messages: []Message{
			{Role: "user", Content: "first"},
			// Earlier turn: an unsigned call (e.g. made by another model)
			// is not validated and must stay unsigned.
			{Role: "assistant", ToolCalls: []ToolCall{call("old1", "ls", `{}`, "")}},
			{Role: "tool", ToolCallID: "old1", Content: "a b"},
			{Role: "assistant", Content: "done"},
			// Two consecutive user messages merge (alternation).
			{Role: "user", Content: "second"},
			{Role: "user", Content: "really"},
			// Current turn, step 1: parallel calls, signature on the first,
			// a Gemini id and a synthetic id, malformed args on the second.
			{Role: "assistant", Content: "checking", Thinking: "summary not replayed", ToolCalls: []ToolCall{
				call("g1", "read", `{"path":"a"}`, "sigA"),
				call(geminiSyntheticIDPrefix+"x", "grep", `{not json`, ""),
			}},
			// Results arrive out of order; they merge into one user content
			// emitted in call order.
			{Role: "tool", ToolCallID: geminiSyntheticIDPrefix + "x", Content: "match"},
			{Role: "tool", ToolCallID: "g1", Content: "contents", Images: []string{"iVBORw0KGgoAAAANSUhEUg=="}},
			// Current turn, step 2: an unsigned call (model switch) gets the dummy.
			{Role: "assistant", ToolCalls: []ToolCall{call("toolu_9", "write", `{"path":"b"}`, ""), call("toolu_10", "write", `{}`, "")}},
			{Role: "tool", ToolCallID: "toolu_9", Content: "ok"},
			{Role: "tool", ToolCallID: "", Content: "ok2"}, // no id: positional match takes the call's id
		},
	})
	jsonEqual(t, req["systemInstruction"], `{"parts":[{"text":"be brief"}]}`)
	jsonEqual(t, req["contents"], `[
		{"role":"user","parts":[{"text":"first"}]},
		{"role":"model","parts":[{"functionCall":{"id":"old1","name":"ls","args":{}}}]},
		{"role":"user","parts":[{"functionResponse":{"id":"old1","name":"ls","response":{"result":"a b"}}}]},
		{"role":"model","parts":[{"text":"done"}]},
		{"role":"user","parts":[{"text":"second"},{"text":"really"}]},
		{"role":"model","parts":[
			{"text":"checking"},
			{"functionCall":{"id":"g1","name":"read","args":{"path":"a"}},"thoughtSignature":"sigA"},
			{"functionCall":{"name":"grep","args":{"_raw":"{not json"}}}
		]},
		{"role":"user","parts":[
			{"functionResponse":{"id":"g1","name":"read","response":{"result":"contents"},
				"parts":[{"inlineData":{"mimeType":"image/png","data":"iVBORw0KGgoAAAANSUhEUg=="}}]}},
			{"functionResponse":{"name":"grep","response":{"result":"match"}}}
		]},
		{"role":"model","parts":[
			{"functionCall":{"id":"toolu_9","name":"write","args":{"path":"b"}},"thoughtSignature":"skip_thought_signature_validator"},
			{"functionCall":{"id":"toolu_10","name":"write","args":{}}}
		]},
		{"role":"user","parts":[
			{"functionResponse":{"id":"toolu_9","name":"write","response":{"result":"ok"}}},
			{"functionResponse":{"id":"toolu_10","name":"write","response":{"result":"ok2"}}}
		]}
	]`)
}

// Results are resolved to unanswered calls of the current batch and take both
// id and name from the call, so id-less parallel calls to the same function
// keep their results paired however the results arrive.
func TestGeminiToolResultPairing(t *testing.T) {
	call := func(id, args string) ToolCall {
		return ToolCall{ID: id, Type: "function", Function: ToolCallFunction{Name: "read", Arguments: args}}
	}
	s1, s2 := geminiSyntheticIDPrefix+"1", geminiSyntheticIDPrefix+"2"
	req := geminiRequestJSON(t, RequestOptions{Model: "m", Messages: []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []ToolCall{call(s1, `{"p":"a"}`), call(s2, `{"p":"b"}`), call("fc_3", `{"p":"c"}`)}},
		{Role: "tool", ToolCallID: s2, Content: "B"},
		{Role: "tool", ToolCallID: "bogus", Content: "first-unanswered"}, // unknown id: positional
		{Role: "tool", ToolCallID: "fc_3", Content: "C"},
	}})
	jsonEqual(t, req["contents"].([]any)[2], `{"role":"user","parts":[
		{"functionResponse":{"name":"read","response":{"result":"first-unanswered"}}},
		{"functionResponse":{"name":"read","response":{"result":"B"}}},
		{"functionResponse":{"id":"fc_3","name":"read","response":{"result":"C"}}}]}`)

	// An unknown result id resolves positionally and is replaced by the
	// call's own wire id.
	req = geminiRequestJSON(t, RequestOptions{Model: "m", Messages: []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "fc_7", Type: "function", Function: ToolCallFunction{Name: "ls", Arguments: "{}"}}}},
		{Role: "tool", ToolCallID: "renamed_by_caller", Content: "x"},
	}})
	jsonEqual(t, req["contents"].([]any)[2], `{"role":"user","parts":[
		{"functionResponse":{"id":"fc_7","name":"ls","response":{"result":"x"}}}]}`)

	// Explicit ids are matched across the whole batch before positional
	// assignment: [no id, id=a] answers b and a, never a twice.
	req = geminiRequestJSON(t, RequestOptions{Model: "m", Messages: []Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "a", Type: "function", Function: ToolCallFunction{Name: "fa", Arguments: "{}"}},
			{ID: "b", Type: "function", Function: ToolCallFunction{Name: "fb", Arguments: "{}"}},
		}},
		{Role: "tool", ToolCallID: "", Content: "for b"},
		{Role: "tool", ToolCallID: "a", Content: "for a"},
		{Role: "tool", ToolCallID: "a", Content: "extra"}, // no call left: goes last, id not repeated
	}})
	jsonEqual(t, req["contents"].([]any)[2], `{"role":"user","parts":[
		{"functionResponse":{"id":"a","name":"fa","response":{"result":"for a"}}},
		{"functionResponse":{"id":"b","name":"fb","response":{"result":"for b"}}},
		{"functionResponse":{"name":"fa","response":{"result":"extra"}}}]}`)
}

func TestGeminiNonStreamingNormalization(t *testing.T) {
	var gotPath, gotKey string
	var gotBody map[string]any
	body := `{"candidates":[{"content":{"role":"model","parts":[
		{"text":"pondering","thought":true},
		{"text":"Hello "},{"text":"world"},
		{"functionCall":{"name":"read","args":{"path":"a"}},"thoughtSignature":"sigA"},
		{"functionCall":{"id":"fc_2","name":"ls"}}
	]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":40,"candidatesTokenCount":20,"thoughtsTokenCount":5,"totalTokenCount":125},
	"modelVersion":"gemini-4-neon-001"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotKey = r.URL.Path, r.Header.Get(GeminiAPIKeyHeader)
		json.NewDecoder(r.Body).Decode(&gotBody)
		io.WriteString(w, body)
	}))
	defer srv.Close()

	resp, err := geminiTestClient(srv.URL).Turn(RequestOptions{Model: "models/gemini-4-neon", Messages: []Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/models/gemini-4-neon:generateContent" || gotKey != "test-key" {
		t.Fatalf("path=%q key=%q", gotPath, gotKey)
	}
	if _, ok := gotBody["contents"]; !ok {
		t.Fatalf("request body missing contents: %v", gotBody)
	}
	msg := resp.Choices[0].Message
	if msg.Role != "assistant" || msg.Content != "Hello world" || msg.Thinking != "pondering" {
		t.Fatalf("message = %+v", msg)
	}
	if len(msg.ToolCalls) != 2 {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	tc0, tc1 := msg.ToolCalls[0], msg.ToolCalls[1]
	if !strings.HasPrefix(tc0.ID, geminiSyntheticIDPrefix) || tc0.Type != "function" || tc0.Function.Name != "read" ||
		tc0.Function.Arguments != `{"path":"a"}` || tc0.ThoughtSignature != "sigA" {
		t.Errorf("call 0 = %+v", tc0)
	}
	if tc1.ID != "fc_2" || tc1.Function.Arguments != "{}" || tc1.ThoughtSignature != "" {
		t.Errorf("call 1 = %+v", tc1)
	}
	if resp.StopReason != "tool_calls" || resp.Choices[0].FinishReason != "tool_calls" || resp.Truncated() {
		t.Errorf("stop = %q/%q", resp.StopReason, resp.Choices[0].FinishReason)
	}
	if resp.Model != "gemini-4-neon-001" {
		t.Errorf("model = %q", resp.Model)
	}
	u := resp.Usage
	if u.PromptTokens != 100 || u.CompletionTokens != 25 || u.TotalTokens != 125 || u.GetCachedTokens() != 40 {
		t.Errorf("usage = %+v", u)
	}

	// The captured call round-trips: replaying it sends the signature and
	// omits the synthesized id.
	replay := geminiRequestJSON(t, RequestOptions{Model: "m", Messages: []Message{
		{Role: "user", Content: "hi"}, msg,
		{Role: "tool", ToolCallID: tc0.ID, Content: "A"},
		{Role: "tool", ToolCallID: tc1.ID, Content: "B"},
	}})
	jsonEqual(t, replay["contents"].([]any)[1], `{"role":"model","parts":[
		{"text":"Hello world"},
		{"functionCall":{"name":"read","args":{"path":"a"}},"thoughtSignature":"sigA"},
		{"functionCall":{"id":"fc_2","name":"ls","args":{}}}]}`)
	jsonEqual(t, replay["contents"].([]any)[2], `{"role":"user","parts":[
		{"functionResponse":{"name":"read","response":{"result":"A"}}},
		{"functionResponse":{"id":"fc_2","name":"ls","response":{"result":"B"}}}]}`)
}

func TestGeminiFinishReasons(t *testing.T) {
	const fc = `{"functionCall":{"id":"fc_1","name":"ls","args":{}}}`
	cases := []struct {
		body, stop string
		truncated  bool
		calls      int
		errStatus  int // non-zero: expect an *APIError with this StatusCode
	}{
		{`{"candidates":[{"content":{"parts":[{"text":"cut"}]},"finishReason":"MAX_TOKENS"}]}`, "length", true, 0, 0},
		{`{"candidates":[{"content":{"parts":[{"text":"x"},` + fc + `]},"finishReason":"MAX_TOKENS"}]}`, "length", true, 1, 0},
		{`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"CONTINUATION"}]}`, "length", true, 0, 0},
		{`{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`, "stop", false, 0, 0},
		{`{"candidates":[{"content":{"parts":[` + fc + `]},"finishReason":"STOP"}]}`, "tool_calls", false, 1, 0},
		{`{"candidates":[{"finishReason":"SAFETY"}]}`, "refusal", false, 0, 0},
		{`{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}]}`, "refusal", false, 0, 0},
		{`{"candidates":[{"finishReason":"ESCALATION"}]}`, "refusal", false, 0, 0},
		{`{"candidates":[{"content":{"parts":[{"text":"?"}]},"finishReason":"OTHER"}]}`, "other", false, 0, 0},
		{`{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":3,"totalTokenCount":3}}`, "refusal", false, 0, 0},
		{`{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad json"}]}`, "", false, 0, 500},
		{`{"candidates":[{"finishReason":"MALFORMED_RESPONSE"}]}`, "", false, 0, 500},
		{`{"candidates":[{"finishReason":"UNEXPECTED_TOOL_CALL"}]}`, "", false, 0, 500},
		{`{"candidates":[{"finishReason":"TOO_MANY_TOOL_CALLS"}]}`, "", false, 0, 500},
		{`{"candidates":[{"finishReason":"MISSING_THOUGHT_SIGNATURE"}]}`, "", false, 0, 400},
		{`{"candidates":[{"finishReason":"PUP_LIMITED_DISABLED"}]}`, "", false, 0, 403},
	}
	for _, tc := range cases {
		var gr geminiResponse
		if err := json.Unmarshal([]byte(tc.body), &gr); err != nil {
			t.Fatal(err)
		}
		resp, err := convertGeminiResponse(&gr, "m")
		if tc.errStatus != 0 {
			ae, ok := AsAPIError(err)
			if !ok || ae.StatusCode != tc.errStatus || resp != nil {
				t.Errorf("%s: want *APIError %d, got resp=%v err=%v", tc.body, tc.errStatus, resp, err)
				continue
			}
			var body struct {
				Error struct{ Message, Status string }
			}
			if json.Unmarshal([]byte(ae.Body), &body) != nil || body.Error.Status == "" ||
				!strings.Contains(body.Error.Message, gr.Candidates[0].FinishReason) {
				t.Errorf("%s: error body %q lacks status/reason", tc.body, ae.Body)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", tc.body, err)
		}
		if resp.StopReason != tc.stop || resp.Choices[0].FinishReason != tc.stop || resp.Truncated() != tc.truncated {
			t.Errorf("%s: stop=%q truncated=%v", tc.body, resp.StopReason, resp.Truncated())
		}
		if n := len(resp.Choices[0].Message.ToolCalls); n != tc.calls {
			t.Errorf("%s: %d tool calls, want %d", tc.body, n, tc.calls)
		}
		if tc.stop == "refusal" && resp.Choices[0].Message.Content != "" {
			t.Errorf("refusal should have empty content")
		}
	}

	// An unspecified block reason with no candidates is not a refusal.
	if _, err := convertGeminiResponse(&geminiResponse{PromptFeedback: &geminiPromptFeedback{BlockReason: "BLOCK_REASON_UNSPECIFIED"}}, "m"); err == nil {
		t.Error("BLOCK_REASON_UNSPECIFIED without candidates should be an error")
	}
}

const geminiStreamBody = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"pondering","thought":true}]}}],"modelVersion":"gemini-4-neon-001"}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"lo"},{"functionCall":{"id":"fc_1","name":"read","args":{"path":"a"}}}]}}],"usageMetadata":{"promptTokenCount":9,"totalTokenCount":9}}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"sigLate"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":16}}

`

func TestGeminiStreaming(t *testing.T) {
	var gotPath, gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotQuery = r.URL.Path, r.URL.RawQuery
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, geminiStreamBody)
	}))
	defer srv.Close()

	var snaps []string
	resp, err := geminiTestClient(srv.URL).TurnStream(RequestOptions{Model: "gemini-4-neon", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(s string) { snaps = append(snaps, s) })
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/models/gemini-4-neon:streamGenerateContent" || gotQuery != "alt=sse" {
		t.Fatalf("path=%q query=%q", gotPath, gotQuery)
	}
	if !reflect.DeepEqual(snaps, []string{"Hel", "Hello"}) {
		t.Fatalf("snapshots = %q", snaps)
	}

	// Equivalent non-streaming response converts to the identical result.
	var gr geminiResponse
	json.Unmarshal([]byte(`{"candidates":[{"content":{"role":"model","parts":[
		{"text":"pondering","thought":true},{"text":"Hello"},
		{"functionCall":{"id":"fc_1","name":"read","args":{"path":"a"}},"thoughtSignature":"sigLate"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":16},
		"modelVersion":"gemini-4-neon-001"}`), &gr)
	want, err := convertGeminiResponse(&gr, "gemini-4-neon")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("streamed != non-streamed\n got: %+v\nwant: %+v", resp, want)
	}
	if tc := resp.Choices[0].Message.ToolCalls; len(tc) != 1 || tc[0].ThoughtSignature != "sigLate" {
		t.Fatalf("late signature not attached: %+v", tc)
	}
	if resp.Usage.CompletionTokens != 6 || resp.Usage.TotalTokens != 16 || resp.StopReason != "tool_calls" {
		t.Fatalf("usage/stop = %+v %q", resp.Usage, resp.StopReason)
	}
}

func TestGeminiStreamTerminal(t *testing.T) {
	partial := `data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}` + "\n\n"
	for _, tc := range []struct {
		name, body string
		complete   bool
	}{
		{"partial EOF", partial, false},
		{"unspecified finish reason", partial + `data: {"candidates":[{"content":{"parts":[{"text":"."}]},"finishReason":"FINISH_REASON_UNSPECIFIED"}]}` + "\n\n", false},
		{"unspecified block reason", `data: {"promptFeedback":{"blockReason":"BLOCK_REASON_UNSPECIFIED"}}` + "\n\n", false},
		{"cut mid-frame", partial + `data: {"candidates":[{"content":{"parts":[{"text":"tr`, false},
		{"finish reason", partial + `data: {"candidates":[{"content":{"parts":[{"text":"."}]},"finishReason":"STOP"}]}` + "\n\n", true},
		{"finish reason, no trailing newline", partial + `data: {"candidates":[{"finishReason":"STOP"}]}`, true},
		{"prompt blocked", `data: {"promptFeedback":{"blockReason":"SAFETY"}}` + "\n\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := assembleGeminiStream(strings.NewReader(tc.body), nil, nil)
			if tc.complete {
				if err != nil || resp == nil {
					t.Fatalf("complete stream: resp=%v err=%v", resp, err)
				}
			} else if resp != nil || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("truncated stream: resp=%v err=%v", resp, err)
			}
		})
	}

	// Errors embedded in the 200 stream surface as *APIError with the
	// embedded code, so callers classify them like HTTP errors.
	const internal = `{"error":{"code":500,"message":"boom","status":"INTERNAL"}}`
	_, err := assembleGeminiStream(strings.NewReader(partial+"data: "+internal+"\n\n"), nil, nil)
	ae, ok := AsAPIError(err)
	if !ok || ae.StatusCode != 500 || ae.Body != internal {
		t.Fatalf("mid-stream error = %v", err)
	}
	_, err = assembleGeminiStream(strings.NewReader(`data: {"error":{"message":"no code"}}`+"\n\n"), nil, nil)
	if ae, ok := AsAPIError(err); !ok || ae.StatusCode != 500 {
		t.Fatalf("code-less embedded error = %v", err)
	}
}

func TestGeminiStreamEmbeddedRateLimit(t *testing.T) {
	const embedded = `{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED",` +
		`"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3s"}]}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Request-Id", "req_9")
		io.WriteString(w, `data: {"candidates":[{"content":{"parts":[{"text":"x"}]}}]}`+"\n\ndata: "+embedded+"\n\n")
	}))
	defer srv.Close()

	_, err := geminiTestClient(srv.URL).TurnStream(RequestOptions{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}}, nil)
	ae, ok := AsAPIError(err)
	if !ok || ae.StatusCode != 429 || !ae.Retryable() || ae.Body != embedded || ae.Header.Get("X-Request-Id") != "req_9" {
		t.Fatalf("err = %#v", err)
	}
	if d, ok := ae.RetryAfter(time.Now()); !ok || d != 3*time.Second {
		t.Fatalf("RetryAfter = %v,%v want 3s", d, ok)
	}
}

func TestGeminiStreamHonorsContext(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"candidates":[{"content":{"parts":[{"text":"x"}]}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	_, err := geminiTestClient(srv.URL).TurnStreamCtx(ctx, RequestOptions{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(string) { cancel() })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestGeminiAPIError(t *testing.T) {
	const body = `{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED",` +
		`"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"17s"}]}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, body)
	}))
	defer srv.Close()

	for _, stream := range []bool{false, true} {
		c := geminiTestClient(srv.URL)
		var err error
		opts := RequestOptions{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}}
		if stream {
			_, err = c.TurnStream(opts, nil)
		} else {
			_, err = c.Turn(opts)
		}
		ae, ok := AsAPIError(err)
		if !ok || ae.StatusCode != 429 || !ae.Retryable() || ae.Body != body {
			t.Fatalf("stream=%v: err = %v", stream, err)
		}
		if d, ok := ae.RetryAfter(time.Now()); !ok || d != 17*time.Second {
			t.Fatalf("RetryAfter = %v,%v want 17s from RetryInfo", d, ok)
		}
	}
}

func TestGeminiListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("path = %q", r.URL.Path)
		}
		switch r.URL.Query().Get("pageToken") {
		case "":
			io.WriteString(w, `{"models":[
				{"name":"models/gemini-4-neon","supportedGenerationMethods":["generateContent","countTokens"]},
				{"name":"models/text-embedding-005","supportedGenerationMethods":["embedContent"]}
			],"nextPageToken":"p2"}`)
		case "p2":
			io.WriteString(w, `{"models":[{"name":"models/gemini-3-flash","supportedGenerationMethods":["generateContent"]}]}`)
		default:
			t.Errorf("unexpected token %q", r.URL.Query().Get("pageToken"))
		}
	}))
	defer srv.Close()

	models, err := geminiTestClient(srv.URL).ListModels()
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range models {
		ids = append(ids, m.ID)
	}
	if !reflect.DeepEqual(ids, []string{"gemini-4-neon", "gemini-3-flash"}) {
		t.Fatalf("ids = %v", ids)
	}
}

func TestGeminiNonStreamingEmbeddedError(t *testing.T) {
	const body = `{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, body) // HTTP 200 carrying an error object
	}))
	defer srv.Close()

	_, err := geminiTestClient(srv.URL).Turn(RequestOptions{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}})
	if ae, ok := AsAPIError(err); !ok || ae.StatusCode != 503 || !ae.Retryable() || ae.Body != body {
		t.Fatalf("err = %v", err)
	}
}

// A refusal-class finish after a streamed functionCall must not leave the call
// dispatchable; text and usage survive.
func TestGeminiRefusalDropsToolCalls(t *testing.T) {
	body := `data: {"candidates":[{"content":{"parts":[{"text":"sure"},{"functionCall":{"id":"fc_1","name":"rm","args":{}}}]}}]}` + "\n\n" +
		`data: {"candidates":[{"finishReason":"SAFETY"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}` + "\n\n"
	gr, err := assembleGeminiStream(strings.NewReader(body), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := convertGeminiResponse(gr, "m")
	if err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if resp.StopReason != "refusal" || len(msg.ToolCalls) != 0 || msg.Content != "sure" || resp.Usage.TotalTokens != 7 {
		t.Fatalf("stop=%q calls=%v content=%q usage=%+v", resp.StopReason, msg.ToolCalls, msg.Content, resp.Usage)
	}
}

type failingReader struct{ err error }

func (r failingReader) Read([]byte) (int, error) { return 0, r.err }

// Only EOF inside a partial frame means a torn stream; any other read error
// (watchdog, cancellation) is preserved.
func TestGeminiStreamPartialFrameReadError(t *testing.T) {
	errStall := errors.New("stream stalled")
	torn := `data: {"candidates":[{"content":{"parts":[{"text":"x"}]}}]}` + "\n\n" + `data: {"candid`
	_, err := assembleGeminiStream(io.MultiReader(strings.NewReader(torn), failingReader{errStall}), nil, nil)
	if !errors.Is(err, errStall) || errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want wrapped errStall", err)
	}
	_, err = assembleGeminiStream(io.MultiReader(strings.NewReader(torn), failingReader{context.Canceled}), nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want wrapped context.Canceled", err)
	}
}

func TestGeminiOnGenerated(t *testing.T) {
	chunks := []struct {
		data string
		fire bool
	}{
		{`{"usageMetadata":{"promptTokenCount":3,"totalTokenCount":3}}`, false},
		{`{"candidates":[{"content":{"parts":[{"text":"hmm","thought":true}]}}]}`, true},
		{`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"s"}]}}]}`, false},
		{`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"ls"}}]}}]}`, true},
		{`{"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}`, true},
	}
	var body strings.Builder
	want := 0
	for _, c := range chunks {
		body.WriteString("data: " + c.data + "\n\n")
		if c.fire {
			want++
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, body.String())
	}))
	defer srv.Close()

	got := 0
	_, err := geminiTestClient(srv.URL).TurnStream(RequestOptions{
		Model: "m", Messages: []Message{{Role: "user", Content: "hi"}},
		OnGenerated: func() { got++ },
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("OnGenerated fired %d times, want %d", got, want)
	}

	// Per chunk: a metadata-only stream never fires it.
	got = 0
	if _, err := assembleGeminiStream(strings.NewReader("data: "+chunks[0].data+"\n\ndata: "+chunks[2].data+"\n\n"), nil, func() { got++ }); err == nil || got != 0 {
		t.Fatalf("metadata-only stream: err=%v fired=%d", err, got)
	}
}
