// Exercise real standard-library signature verification and malformed inputs.
package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

func TestSignedMessageAndTampering(t *testing.T) {
	key, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	message := []byte("synthetic review context")
	packet := append(append(append([]byte{}, key...), ed25519.Sign(private, message)...), message...)
	if !verify(bytes.NewReader(packet)) {
		t.Fatal("valid signature rejected")
	}
	for _, offset := range []int{0, 32, 96} {
		changed := append([]byte{}, packet...)
		changed[offset] ^= 1
		if verify(bytes.NewReader(changed)) {
			t.Fatalf("changed byte %d accepted", offset)
		}
	}
	for length := 0; length <= 96; length++ {
		if verify(bytes.NewReader(packet[:length])) {
			t.Fatalf("truncated packet of length %d accepted", length)
		}
	}
}

func TestRFC8032IndependentVector(t *testing.T) {
	// RFC 8032 section 7.1, TEST 2: one-byte message, independently specified.
	decode := func(value string) []byte {
		result, err := hex.DecodeString(value)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	key := decode("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
	signature := decode("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00")
	packet := append(append(key, signature...), 0x72)
	if !verify(bytes.NewReader(packet)) {
		t.Fatal("published RFC vector rejected")
	}
}
