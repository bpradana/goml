package main

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/md5"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
)

const (
	datasetURL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	datasetMD5 = "c32a1d4ab5d03f1284b67883e8d87530"
)

var batchNames = []string{"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"}

func downloadDataset(ctx context.Context, destination, url, checksum string) error {
	if _, err := os.Stat(destination); err == nil {
		return fmt.Errorf("dataset directory already exists: %s (use train, or choose another -data directory)", destination)
	} else if !os.IsNotExist(err) {
		return err
	}

	parent := filepath.Dir(destination)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}

	staging, err := os.MkdirTemp(parent, ".cifar-download-")
	if err != nil {
		return err
	}

	defer func() { _ = os.RemoveAll(staging) }()

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return fmt.Errorf("download CIFAR-10: %w", err)
	}

	defer func() { _ = response.Body.Close() }()

	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download CIFAR-10: HTTP %s", response.Status)
	}

	archive, err := os.CreateTemp(staging, "archive-*.tar.gz")
	if err != nil {
		return err
	}
	defer func() { _ = archive.Close() }()

	// MD5 matches the checksum published by the dataset's authors.
	digest := md5.New()

	const maxArchive = 180 << 20

	written, err := io.Copy(io.MultiWriter(archive, digest), io.LimitReader(response.Body, maxArchive+1))
	if err != nil {
		return err
	}

	if written > maxArchive || fmt.Sprintf("%x", digest.Sum(nil)) != checksum {
		return fmt.Errorf("CIFAR-10 archive size/checksum mismatch")
	}

	if _, err := archive.Seek(0, io.SeekStart); err != nil {
		return err
	}

	extracted := filepath.Join(staging, "extracted")
	if err := os.Mkdir(extracted, 0o755); err != nil {
		return err
	}

	if err := extractDataset(archive, extracted); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	return os.Rename(extracted, destination)
}

func extractDataset(source io.Reader, destination string) error {
	gz, err := gzip.NewReader(source)
	if err != nil {
		return err
	}
	defer func() { _ = gz.Close() }()

	reader := tar.NewReader(gz)
	seen := make(map[string]bool)

	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}

		if err != nil {
			return err
		}

		if path.IsAbs(header.Name) || path.Clean(header.Name) != strings.TrimSuffix(header.Name, "/") || strings.Contains(header.Name, "\\") {
			return fmt.Errorf("unsafe archive path %q", header.Name)
		}

		if header.Typeflag == tar.TypeDir && header.Name == "cifar-10-batches-bin/" {
			continue
		}

		if header.Typeflag != tar.TypeReg || path.Dir(header.Name) != "cifar-10-batches-bin" {
			return fmt.Errorf("unexpected archive entry %q", header.Name)
		}

		name := path.Base(header.Name)
		if name == "batches.meta.txt" || name == "readme.html" {
			if header.Size > 1<<20 {
				return fmt.Errorf("oversized archive metadata")
			}

			continue
		}

		known := false
		for _, expected := range batchNames {
			known = known || name == expected
		}

		if !known || seen[name] || header.Size <= 0 || header.Size > 10000*recordSize || header.Size%recordSize != 0 {
			return fmt.Errorf("unexpected, duplicate, or malformed batch %q", header.Name)
		}

		contents, err := io.ReadAll(reader)
		if err != nil {
			return err
		}

		if err := validateRecords(contents); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}

		if err := os.WriteFile(filepath.Join(destination, name), contents, 0o644); err != nil {
			return err
		}

		seen[name] = true
	}

	if len(seen) != len(batchNames) {
		return fmt.Errorf("archive is missing CIFAR-10 batches")
	}
	// Read the gzip trailer too; tar may finish before the compressed stream.
	_, err = io.Copy(io.Discard, gz)

	return err
}
