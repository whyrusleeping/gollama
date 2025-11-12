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
