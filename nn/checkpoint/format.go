package checkpoint

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

const magic = "GOMLCP01"

const headerSize = len(magic) + 8 + sha256.Size

const maxPayloadSize = 64 << 20

type document struct {
	Kind         string          `json:"kind"`
	Architecture json.RawMessage `json:"architecture,omitempty"`
	Parameters   []storedTensor  `json:"parameters"`
}

type storedTensor struct {
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  []byte `json:"data"`
}

func encodeDocument(state document) ([]byte, error) {
	payload, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("checkpoint: encode: %w", err)
	}

	if len(payload) > maxPayloadSize {
		return nil, fmt.Errorf("checkpoint: payload exceeds %d bytes", maxPayloadSize)
	}

	result := make([]byte, headerSize+len(payload))
	copy(result, magic)
	binary.LittleEndian.PutUint64(result[len(magic):], uint64(len(payload)))
	digest := sha256.Sum256(payload)
	copy(result[len(magic)+8:], digest[:])
	copy(result[headerSize:], payload)

	return result, nil
}

func decodeDocument(reader io.Reader) (document, error) {
	var (
		state  document
		header [headerSize]byte
	)
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return state, fmt.Errorf("checkpoint: header: %w", err)
	}

	if string(header[:len(magic)]) != magic {
		return state, fmt.Errorf("checkpoint: unsupported format or version")
	}

	size := binary.LittleEndian.Uint64(header[len(magic):])
	if size == 0 || size > maxPayloadSize {
		return state, fmt.Errorf("checkpoint: invalid payload size %d", size)
	}

	payload, err := io.ReadAll(io.LimitReader(reader, int64(size)+1))
	if err != nil {
		return state, fmt.Errorf("checkpoint: payload: %w", err)
	}

	if len(payload) < int(size) {
		return state, fmt.Errorf("checkpoint: payload: %w", io.ErrUnexpectedEOF)
	}

	if len(payload) > int(size) {
		return state, fmt.Errorf("checkpoint: trailing bytes")
	}

	digest := sha256.Sum256(payload)
	if !bytes.Equal(header[len(magic)+8:], digest[:]) {
		return state, fmt.Errorf("checkpoint: checksum mismatch")
	}

	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&state); err != nil {
		return state, fmt.Errorf("checkpoint: metadata: %w", err)
	}

	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return state, fmt.Errorf("checkpoint: trailing metadata")
	}

	if state.Kind != "weights" && state.Kind != "network" {
		return state, fmt.Errorf("checkpoint: unknown kind %q", state.Kind)
	}

	if state.Kind == "weights" && len(state.Architecture) != 0 {
		return state, fmt.Errorf("checkpoint: weights file cannot include architecture")
	}

	if state.Kind == "network" && len(state.Architecture) == 0 {
		return state, fmt.Errorf("checkpoint: network file requires architecture")
	}

	return state, nil
}
