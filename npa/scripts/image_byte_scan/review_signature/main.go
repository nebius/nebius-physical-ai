// Verify one externally authorized Ed25519 review without exposing its contents.
package main

import (
	"crypto/ed25519"
	"io"
	"os"
)

func verify(input io.Reader) bool {
	key := make([]byte, ed25519.PublicKeySize)
	signature := make([]byte, ed25519.SignatureSize)
	if _, err := io.ReadFull(input, key); err != nil {
		return false
	}
	if _, err := io.ReadFull(input, signature); err != nil {
		return false
	}
	message, err := io.ReadAll(input)
	return err == nil && len(message) > 0 && ed25519.Verify(key, message, signature)
}

func main() {
	if len(os.Args) != 1 || !verify(os.Stdin) {
		os.Exit(1)
	}
}
