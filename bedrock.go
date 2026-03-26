package gollama

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// Bedrock model inference profile IDs.
const (
	BedrockHaiku45  = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
	BedrockSonnet46 = "global.anthropic.claude-sonnet-4-6"
	BedrockOpus46   = "global.anthropic.claude-opus-4-6-v1"
)

// BedrockConfig holds AWS credentials and region for Bedrock API access.
type BedrockConfig struct {
	Region    string
	AccessKey string
	SecretKey string
	Token     string // optional session token for temporary credentials
}

// bedrockRequest is the Bedrock variant of the Anthropic request.
// It omits the model field (model is specified in the URL path) and includes anthropic_version.
type bedrockRequest struct {
	AnthropicVersion string                 `json:"anthropic_version"`
	MaxTokens        int                    `json:"max_tokens"`
	System           []anthropicSystemBlock `json:"system,omitempty"`
	Messages         []anthropicMessage     `json:"messages"`
	Tools            []anthropicTool        `json:"tools,omitempty"`
}

// NewBedrockClient creates a new client configured for AWS Bedrock.
// The region determines the endpoint URL. Model is specified per-request in RequestOptions.Model
// using Bedrock model IDs (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0").
func NewBedrockClient(region, accessKey, secretKey, sessionToken string) *Client {
	c := NewClient(fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", region))
	c.bedrock = &BedrockConfig{
		Region:    region,
		AccessKey: accessKey,
		SecretKey: secretKey,
		Token:     sessionToken,
	}
	return c
}

// SetAWSAuth configures AWS credentials for Bedrock API access.
// This can be used with a client created via NewClient with a Bedrock endpoint URL.
func (c *Client) SetAWSAuth(region, accessKey, secretKey, sessionToken string) {
	c.bedrock = &BedrockConfig{
		Region:    region,
		AccessKey: accessKey,
		SecretKey: secretKey,
		Token:     sessionToken,
	}
}

// IsBedrockAPI checks if the client is configured to use AWS Bedrock.
func (c *Client) IsBedrockAPI() bool {
	return c.bedrock != nil
}

// ChatCompletionBedrock sends a request using AWS Bedrock's invoke model endpoint.
// It reuses the Anthropic request/response format, signing requests with AWS Signature V4.
func (c *Client) ChatCompletionBedrock(opts RequestOptions) (*ResponseMessageGenerate, error) {
	antReq, err := buildAnthropicRequest(opts)
	if err != nil {
		return nil, fmt.Errorf("error building request: %w", err)
	}

	// Convert to Bedrock request (no model field, add anthropic_version)
	req := bedrockRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        antReq.MaxTokens,
		System:           antReq.System,
		Messages:         antReq.Messages,
		Tools:            antReq.Tools,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	// Construct the invoke model endpoint URL.
	// Bedrock model IDs may contain colons (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
	// which must be percent-encoded in the URL path.
	modelID := url.PathEscape(opts.Model)
	endpoint := fmt.Sprintf("/model/%s/invoke", modelID)
	fullURL := c.baseURL + endpoint

	maxRetries := 5
	baseDelay := 5 * time.Second

	for attempt := 0; attempt <= maxRetries; attempt++ {
		httpReq, err := http.NewRequest("POST", fullURL, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("error creating request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "application/json")

		if err := c.signRequest(httpReq, body); err != nil {
			return nil, fmt.Errorf("error signing request: %w", err)
		}

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("error sending request: %w", err)
		}

		if resp.StatusCode == http.StatusOK {
			defer resp.Body.Close()
			return parseAnthropicResponse(resp)
		}

		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if isRetryableStatus(resp.StatusCode) && attempt < maxRetries {
			delay := baseDelay * time.Duration(1<<attempt)
			log.Printf("Bedrock API returned %d, retrying in %v (attempt %d/%d)", resp.StatusCode, delay, attempt+1, maxRetries)
			time.Sleep(delay)
			continue
		}

		return nil, fmt.Errorf("Bedrock API returned non-200 status code %d (url: %s): %s", resp.StatusCode, fullURL, string(bodyBytes))
	}

	return nil, fmt.Errorf("max retries exceeded")
}

// signRequest signs an HTTP request using AWS Signature Version 4.
func (c *Client) signRequest(req *http.Request, payload []byte) error {
	return signRequestWithTime(c.bedrock, req, payload, time.Now().UTC())
}

// signRequestWithTime signs an HTTP request using AWS Signature Version 4 with an explicit timestamp.
// Extracted for testability.
func signRequestWithTime(cfg *BedrockConfig, req *http.Request, payload []byte, now time.Time) error {
	datestamp := now.Format("20060102")
	amzdate := now.Format("20060102T150405Z")
	service := "bedrock"

	req.Header.Set("X-Amz-Date", amzdate)
	if cfg.Token != "" {
		req.Header.Set("X-Amz-Security-Token", cfg.Token)
	}

	payloadHash := sha256Hex(payload)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	// Build canonical headers — must include host, content-type, and all x-amz-* headers
	host := req.URL.Host
	canonicalHeaders := map[string]string{
		"content-type":         req.Header.Get("Content-Type"),
		"host":                 host,
		"x-amz-content-sha256": payloadHash,
		"x-amz-date":           amzdate,
	}
	if cfg.Token != "" {
		canonicalHeaders["x-amz-security-token"] = cfg.Token
	}

	var signedHeaderKeys []string
	for k := range canonicalHeaders {
		signedHeaderKeys = append(signedHeaderKeys, k)
	}
	sort.Strings(signedHeaderKeys)

	var canonicalHeaderStr strings.Builder
	for _, k := range signedHeaderKeys {
		canonicalHeaderStr.WriteString(k)
		canonicalHeaderStr.WriteString(":")
		canonicalHeaderStr.WriteString(canonicalHeaders[k])
		canonicalHeaderStr.WriteString("\n")
	}
	signedHeaders := strings.Join(signedHeaderKeys, ";")

	// Build canonical request.
	// AWS SigV4 requires stricter URI encoding than RFC 3986 — everything except
	// A-Za-z0-9-._~ must be percent-encoded in each path segment.
	canonicalURI := awsCanonicalURI(req.URL.Path)
	canonicalQueryString := req.URL.RawQuery

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQueryString,
		canonicalHeaderStr.String(),
		signedHeaders,
		payloadHash,
	}, "\n")

	// Build string to sign
	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", datestamp, cfg.Region, service)
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzdate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	// Derive signing key and calculate signature
	signingKey := deriveSigningKey(cfg.SecretKey, datestamp, cfg.Region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	// Set Authorization header
	authHeader := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		cfg.AccessKey, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", authHeader)

	return nil
}

// awsCanonicalURI builds the canonical URI for AWS SigV4 signing.
// Each path segment is encoded with AWS's strict rules where only
// A-Za-z0-9-._~ are left unencoded.
func awsCanonicalURI(path string) string {
	if path == "" {
		return "/"
	}
	segments := strings.Split(path, "/")
	for i, seg := range segments {
		segments[i] = awsURIEncode(seg)
	}
	return strings.Join(segments, "/")
}

// awsURIEncode percent-encodes a string using AWS SigV4 rules.
// Only unreserved characters (A-Za-z0-9-._~) are left unencoded.
func awsURIEncode(s string) string {
	var buf strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
			c == '-' || c == '.' || c == '_' || c == '~' {
			buf.WriteByte(c)
		} else {
			fmt.Fprintf(&buf, "%%%02X", c)
		}
	}
	return buf.String()
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func deriveSigningKey(secret, datestamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), []byte(datestamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return kSigning
}
