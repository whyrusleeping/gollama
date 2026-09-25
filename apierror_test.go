package gollama

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestAPIErrorPreservesMetadataAndLegacyText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.Header().Set("X-Request-Id", "req_1")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":"slow down"}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	c.SetMaxRetries(0)
	c.SetAPIKey("secret-key")
	_, err := c.prepareRequestCtx(context.Background(), map[string]any{}, "/x")
	if err == nil {
		t.Fatal("expected error")
	}
	want := `API returned non-200 status code 429: {"error":"slow down"}`
	if err.Error() != want {
		t.Fatalf("Error() = %q, want %q", err.Error(), want)
	}
	ae, ok := AsAPIError(fmt.Errorf("wrapped: %w", err))
	if !ok {
		t.Fatal("AsAPIError failed through wrapping")
	}
	if ae.StatusCode != 429 || !ae.Retryable() || ae.Header.Get("X-Request-Id") != "req_1" {
		t.Fatalf("unexpected APIError: %+v", ae)
	}
	if ae.Header.Get("x-api-key") != "" {
		t.Fatal("request credentials leaked into APIError headers")
	}
	if d, ok := ae.RetryAfter(time.Now()); !ok || d != 7*time.Second {
		t.Fatalf("RetryAfter = %v,%v", d, ok)
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		in   string
		want time.Duration
		ok   bool
	}{
		{"", 0, false},
		{"0", 0, true},
		{"120", 120 * time.Second, true},
		{"-5", 0, false},
		{"soon", 0, false},
		{"99999999999999999999", 0, false},
		{now.Add(30 * time.Second).Format(http.TimeFormat), 30 * time.Second, true},
		{now.Add(-time.Hour).Format(http.TimeFormat), 0, true},
	}
	for _, tc := range cases {
		got, ok := parseRetryAfter(tc.in, now)
		if got != tc.want || ok != tc.ok {
			t.Errorf("parseRetryAfter(%q) = %v,%v want %v,%v", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

func TestRetryRingHonoursRetryAfter(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, "ok")
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	c.SetMaxRetries(1)
	start := time.Now()
	resp, err := c.prepareRequestCtx(context.Background(), map[string]any{}, "/x")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	// Without Retry-After the first backoff would be baseDelay (5s).
	if el := time.Since(start); el > 2*time.Second {
		t.Fatalf("retry took %v; Retry-After: 0 not honoured", el)
	}
}

func TestSetHTTPClientRemovesTotalTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fl := w.(http.Flusher)
		for i := 0; i < 3; i++ {
			fmt.Fprint(w, "chunk\n")
			fl.Flush()
			time.Sleep(150 * time.Millisecond)
		}
	}))
	defer srv.Close()

	c := NewClient(srv.URL)
	// A tight total timeout cuts off a progressing stream...
	c.SetHTTPClient(&http.Client{Timeout: 200 * time.Millisecond})
	resp, err := c.prepareGetCtx(context.Background(), "/s")
	if err == nil {
		_, err = readAllErr(resp)
	}
	if err == nil {
		t.Fatal("expected total timeout to cut the stream")
	}
	// ...while a caller-supplied client without one lets it finish.
	c.SetHTTPClient(&http.Client{})
	resp, err = c.prepareGetCtx(context.Background(), "/s")
	if err != nil {
		t.Fatal(err)
	}
	body, err := readAllErr(resp)
	if err != nil || strings.Count(body, "chunk") != 3 {
		t.Fatalf("body=%q err=%v", body, err)
	}
	// nil restores the default client.
	c.SetHTTPClient(nil)
	if c.httpClient == nil || c.httpClient.Timeout != defaultHTTPTimeout {
		t.Fatal("SetHTTPClient(nil) did not restore default")
	}
}

func readAllErr(resp *http.Response) (string, error) {
	defer resp.Body.Close()
	var sb strings.Builder
	buf := make([]byte, 64)
	for {
		n, err := resp.Body.Read(buf)
		sb.Write(buf[:n])
		if errors.Is(err, io.EOF) {
			return sb.String(), nil
		}
		if err != nil {
			return sb.String(), err
		}
	}
}
