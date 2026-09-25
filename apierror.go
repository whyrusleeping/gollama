package gollama

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// APIError is returned when a provider responds with a non-200 HTTP status.
// It preserves the transport metadata callers need for structured
// classification and retry scheduling. Its Error string is unchanged from the
// historical format ("API returned non-200 status code NNN: body") so textual
// classifiers keep working.
type APIError struct {
	// StatusCode is the HTTP status code.
	StatusCode int
	// Body is the raw response body (may be empty).
	Body string
	// Header holds the response headers (a clone; never the request headers,
	// so it does not contain credentials sent by the client).
	Header http.Header
}

func newAPIError(resp *http.Response, body []byte) *APIError {
	return &APIError{
		StatusCode: resp.StatusCode,
		Body:       string(body),
		Header:     resp.Header.Clone(),
	}
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API returned non-200 status code %d: %s", e.StatusCode, e.Body)
}

// Retryable reports whether the status is one the built-in retry ring treats
// as transient (429, 503, 529).
func (e *APIError) Retryable() bool { return isRetryableStatus(e.StatusCode) }

// RetryAfter parses the standard Retry-After response header (delta-seconds
// or HTTP-date) relative to now. It returns ok=false when the header is
// absent or malformed. A date in the past yields (0, true). Callers are
// responsible for bounding the result.
func (e *APIError) RetryAfter(now time.Time) (time.Duration, bool) {
	if e == nil || e.Header == nil {
		return 0, false
	}
	return parseRetryAfter(e.Header.Get("Retry-After"), now)
}

func parseRetryAfter(v string, now time.Time) (time.Duration, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.ParseInt(v, 10, 64); err == nil {
		if secs < 0 {
			return 0, false
		}
		if secs > int64((1<<63-1)/int64(time.Second)) {
			return 0, false
		}
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(v); err == nil {
		d := t.Sub(now)
		if d < 0 {
			d = 0
		}
		return d, true
	}
	return 0, false
}

// AsAPIError unwraps err to an *APIError if one is present in its chain.
func AsAPIError(err error) (*APIError, bool) {
	var ae *APIError
	if errors.As(err, &ae) {
		return ae, true
	}
	return nil, false
}
