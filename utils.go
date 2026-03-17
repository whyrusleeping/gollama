package gollama

import (
	"encoding/base64"
	"fmt"
	"os"
)

// AddImageFromFile encodes an image file as base64 and returns the encoded string.
// This string can be added to Message.Images for sending images to the API.
func AddImageFromFile(filename string) (string, error) {
	// Read the file and encode as base64
	imageBytes, err := os.ReadFile(filename)
	if err != nil {
		return "", fmt.Errorf("error reading image file: %w", err)
	}

	return base64.StdEncoding.EncodeToString(imageBytes), nil
}

// DetectImageMediaType sniffs the media type from base64-encoded image data.
// Falls back to "image/jpeg" if the format is unrecognized.
func DetectImageMediaType(b64data string) string {
	// Decode enough bytes to check magic numbers (16 bytes is plenty)
	// Use RawStdEncoding-compatible decoding by trying standard first
	raw, err := base64.StdEncoding.DecodeString(b64data[:min(24, len(b64data))])
	if err != nil || len(raw) < 4 {
		return "image/jpeg"
	}

	switch {
	case len(raw) >= 3 && raw[0] == 0xFF && raw[1] == 0xD8 && raw[2] == 0xFF:
		return "image/jpeg"
	case len(raw) >= 4 && raw[0] == 0x89 && raw[1] == 'P' && raw[2] == 'N' && raw[3] == 'G':
		return "image/png"
	case len(raw) >= 4 && raw[0] == 'G' && raw[1] == 'I' && raw[2] == 'F' && raw[3] == '8':
		return "image/gif"
	case len(raw) >= 12 && raw[0] == 'R' && raw[1] == 'I' && raw[2] == 'F' && raw[3] == 'F' &&
		raw[8] == 'W' && raw[9] == 'E' && raw[10] == 'B' && raw[11] == 'P':
		return "image/webp"
	default:
		return "image/jpeg"
	}
}
