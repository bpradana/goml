package checkpoint

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"testing"
)

func TestDecodeDocument_RejectsCorruptionTruncationVersionsAndTrailingBytes(t *testing.T) {
	valid, err := encodeDocument(document{Kind: "weights"})
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name   string
		mutate func([]byte) []byte
	}{
		{"TruncatedHeader", func(data []byte) []byte { return data[:4] }},
		{"TruncatedPayload", func(data []byte) []byte { return data[:len(data)-1] }},
		{"Version", func(data []byte) []byte { data[7] = '2'; return data }},
		{"Checksum", func(data []byte) []byte { data[headerSize] ^= 1; return data }},
		{"TrailingBytes", func(data []byte) []byte { return append(data, 0) }},
		{"OversizedHeader", func(data []byte) []byte {
			binary.LittleEndian.PutUint64(data[len(magic):], maxPayloadSize+1)
			return data
		}},
		{"ZeroSize", func(data []byte) []byte { binary.LittleEndian.PutUint64(data[len(magic):], 0); return data }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			data := tt.mutate(bytes.Clone(valid))
			if _, err := decodeDocument(bytes.NewReader(data)); err == nil {
				t.Fatal("accepted invalid checkpoint")
			}
		})
	}
}

func TestDecodeDocument_RejectsWrongKindsAndMissingArchitecture(t *testing.T) {
	for _, state := range []document{
		{Kind: "unknown"},
		{Kind: "network"},
		{Kind: "weights", Architecture: json.RawMessage(`{}`)},
	} {
		data, err := encodeDocument(state)
		if err != nil {
			t.Fatal(err)
		}

		if _, err := decodeDocument(bytes.NewReader(data)); err == nil {
			t.Fatal("accepted inconsistent kind or architecture")
		}
	}
}

func TestDecodeDocument_HugeDeclaredSizeDoesNotRequireHugeInput(t *testing.T) {
	header := make([]byte, headerSize)
	copy(header, magic)
	binary.LittleEndian.PutUint64(header[len(magic):], maxPayloadSize)

	if _, err := decodeDocument(bytes.NewReader(header)); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected truncated payload error, got %v", err)
	}
}

func FuzzUnmarshal(f *testing.F) {
	weights, err := encodeDocument(document{Kind: "weights"})
	if err != nil {
		f.Fatal(err)
	}

	network, err := encodeDocument(document{Kind: "network", Architecture: json.RawMessage(`{"root":0,"layers":[{"type":"sequential"}]}`)})
	if err != nil {
		f.Fatal(err)
	}

	f.Add(weights)
	f.Add(network)
	f.Add([]byte("invalid"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 32768 {
			return
		}

		_, _ = Unmarshal(data)
	})
}
