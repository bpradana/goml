package checkpoint

import (
	"fmt"
	"os"
	"path/filepath"
)

func writeFile(path string, data []byte) error {
	if path == "" {
		return fmt.Errorf("checkpoint: empty destination path")
	}

	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return fmt.Errorf("checkpoint: create directory: %w", err)
	}

	file, err := os.CreateTemp(directory, ".goml-checkpoint-*")
	if err != nil {
		return fmt.Errorf("checkpoint: create temporary file: %w", err)
	}
	defer func() { _ = os.Remove(file.Name()) }()
	defer func() { _ = file.Close() }()

	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("checkpoint: write: %w", err)
	}

	if err := file.Sync(); err != nil {
		return fmt.Errorf("checkpoint: sync: %w", err)
	}

	if err := file.Close(); err != nil {
		return fmt.Errorf("checkpoint: close: %w", err)
	}

	if err := os.Rename(file.Name(), path); err != nil {
		return fmt.Errorf("checkpoint: replace destination: %w", err)
	}

	return nil
}

func readFile(path string) (document, error) {
	file, err := os.Open(path)
	if err != nil {
		return document{}, fmt.Errorf("checkpoint: open: %w", err)
	}
	defer func() { _ = file.Close() }()

	return decodeDocument(file)
}
