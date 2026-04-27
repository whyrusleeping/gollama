//go:build ignore

// Generates testdata/secret.pdf — a minimal hand-rolled PDF used by
// pdf_test.go to verify that document blocks in tool results round-trip
// through the Anthropic API. Run with:
//
//	go run testdata/genpdf.go
package main

import (
	"bytes"
	"fmt"
	"os"
)

// The test asserts the model can extract this exact phrase from the PDF.
// Pick something rare enough that it can't be guessed without reading the doc.
const secretLine = "The secret password is XYZZY42-NOVEMBER-CHESHIRE."

func main() {
	pdf, err := buildPDF(secretLine)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build pdf:", err)
		os.Exit(1)
	}
	if err := os.WriteFile("testdata/secret.pdf", pdf, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote testdata/secret.pdf (%d bytes)\n", len(pdf))
}

// buildPDF emits a minimal PDF 1.4 document containing a single page with
// `text` rendered in Helvetica. Keeps the byte layout deterministic so the
// test fixture is stable across runs.
func buildPDF(text string) ([]byte, error) {
	var buf bytes.Buffer
	offsets := make([]int, 6) // index 1..5 used; 0 is the free entry

	// Header — the binary marker keeps tools from treating the file as ASCII.
	buf.WriteString("%PDF-1.4\n")
	buf.Write([]byte{'%', 0xE2, 0xE3, 0xCF, 0xD3, '\n'})

	writeObj := func(idx int, body string) {
		offsets[idx] = buf.Len()
		fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", idx, body)
	}

	writeObj(1, "<< /Type /Catalog /Pages 2 0 R >>")
	writeObj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
	writeObj(3, "<< /Type /Page /Parent 2 0 R "+
		"/Resources << /Font << /F1 4 0 R >> >> "+
		"/MediaBox [0 0 612 792] /Contents 5 0 R >>")
	writeObj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

	stream := fmt.Sprintf("BT /F1 18 Tf 72 720 Td (%s) Tj ET\n", escapePDFString(text))
	writeObj(5, fmt.Sprintf("<< /Length %d >>\nstream\n%sendstream", len(stream), stream))

	xrefOffset := buf.Len()
	buf.WriteString("xref\n0 6\n")
	buf.WriteString("0000000000 65535 f \n")
	for i := 1; i <= 5; i++ {
		fmt.Fprintf(&buf, "%010d 00000 n \n", offsets[i])
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", xrefOffset)

	return buf.Bytes(), nil
}

// escapePDFString escapes characters that would prematurely terminate or
// break a PDF literal string (parens and backslash).
func escapePDFString(s string) string {
	var out bytes.Buffer
	for _, c := range s {
		switch c {
		case '(', ')', '\\':
			out.WriteByte('\\')
		}
		out.WriteRune(c)
	}
	return out.String()
}
